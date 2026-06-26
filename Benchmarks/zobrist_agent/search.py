from referee.game import PlayerColor, Board, Action, \
    PlaceAction, MoveAction, EatAction, CascadeAction, Coord
from .actions import legal_play_actions
from .evaluate import evaluate, INF
from .game_state import COORD_TABLE
import time
import random

MATE = 1_000_000 # score used for forced wins (offset by ply to prefer shorter mates)
NODE_CHECK = 4096 # check the clock every N nodes
MAX_PLY = 64 # maximum search depth / killer-table size
TT_MAX_ENTRIES = 1_000_000 # evict oldest entry when this is reached

# Transposition table flag values
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

# Pre-built Coord cache so hot paths never call Coord(r, c) repeatedly
_COORD_CACHE: dict[tuple[int, int], Coord] = {
    (r, c): Coord(r, c)
    for r in range(8) for c in range(8)
}


class SearchAborted(Exception):
    """Raised when the search deadline is exceeded."""
    pass


class TranspositionTable:
    """Zobrist-hashed transposition table with soft aging."""

    def __init__(self):
        self.table: dict = {}
        self.zobrist_table = self._init_zobrist()

    def _init_zobrist(self) -> dict:
        """Build random 64-bit keys for every (coord, color, height) triple plus side-to-move."""
        random.seed(42)
        zobrist = {}
        for coord in COORD_TABLE:
            for color in (PlayerColor.RED, PlayerColor.BLUE):
                for height in range(1, 65):
                    zobrist[(coord, color, height)] = random.getrandbits(64)
        zobrist['red_turn'] = random.getrandbits(64)
        zobrist['blue_turn'] = random.getrandbits(64)
        return zobrist

    def get_hash(self, board: Board) -> int:
        """XOR-hash of all occupied cells plus the side-to-move key."""
        h = 0
        zt = self.zobrist_table
        for coord in COORD_TABLE:
            cell = board[coord]
            if cell.is_stack:
                h ^= zt[(coord, cell.color, cell.height)]
        h ^= zt['red_turn'] if board._turn_color == PlayerColor.RED else zt['blue_turn']
        return h

    def store(self, h: int, depth: int, score: float, flag: int, best_action: Action = None):
        """Store an entry; skip if a deeper entry already exists."""
        entry = self.table.get(h)
        if entry is not None and entry[0] > depth:
            return
        if len(self.table) >= TT_MAX_ENTRIES and h not in self.table:
            self.table.pop(next(iter(self.table))) # evict oldest (insertion-order)
        self.table[h] = (depth, score, flag, best_action)

    def soft_clear(self):
        """Age entries by reducing depth by 2; trim table if over half capacity."""
        if len(self.table) > TT_MAX_ENTRIES // 2:
            self.table = {
                h: (max(0, d - 2), s, f, m)
                for h, (d, s, f, m) in self.table.items()
            }

    def lookup(self, h: int, depth: int, alpha: float, beta: float) -> tuple[float | None, Action | None]:
        """Return (score, hash_move). score is None when no cutoff can be made."""
        entry = self.table.get(h)
        if entry is None:
            return None, None
        tt_depth, tt_score, tt_flag, tt_move = entry
        if tt_depth < depth:
            return None, tt_move # too shallow for pruning but hash_move still useful
        if tt_flag == TT_EXACT:
            return tt_score, tt_move
        if tt_flag == TT_LOWER and tt_score >= beta:
            return tt_score, tt_move
        if tt_flag == TT_UPPER and tt_score <= alpha:
            return tt_score, tt_move
        return None, tt_move


def _score_action(board: Board, action: Action, hash_move: Action | None,
                  killers: list, history: dict) -> int:
    """Move-ordering score; higher = searched earlier."""
    if action == hash_move:
        return 10_000_000

    if isinstance(action, EatAction):
        victim_cell = board[action.coord + action.direction]
        return 1_000_000 + victim_cell.height * 100

    if isinstance(action, CascadeAction):
        threat = _cascade_threat_sum(board, action, board._turn_color.opponent)
        return 500_000 + threat * 50

    if isinstance(action, MoveAction):
        dest_cell = board[action.coord + action.direction]
        is_merge = (not dest_cell.is_empty) and (dest_cell.color == board._turn_color)
        base = 200_000 if is_merge else 0
        k1, k2 = (killers[0], killers[1]) if killers else (None, None)
        if action == k1:
            base += 90_000
        elif action == k2:
            base += 80_000
        base += history.get(_history_key(action), 0)
        return base

    return 0


def _order_moves(board: Board, actions: list[Action], hash_move: Action | None,
                 killers: list, history: dict) -> None:
    """Sort actions in-place by descending move-ordering score."""
    actions.sort(key=lambda a: _score_action(board, a, hash_move, killers, history), reverse=True)


def _history_key(action: Action) -> tuple:
    """Compact tuple key for the history heuristic table."""
    if isinstance(action, (MoveAction, EatAction, CascadeAction)):
        return (action.__class__.__name__,
                action.coord.r, action.coord.c,
                action.direction.r, action.direction.c)
    return ('P', action.coord.r, action.coord.c)


def _record_cutoff(action: Action, depth: int, ply: int,
                   killers: list[list], history: dict) -> None:
    """Update killer and history tables on a beta cutoff."""
    if isinstance(action, EatAction):
        return  # captures are already ordered by MVV; skip
    kls = killers[ply] if 0 <= ply < len(killers) else None
    if kls is not None and kls[0] != action:
        kls[1] = kls[0]
        kls[0] = action
    key = _history_key(action)
    history[key] = history.get(key, 0) + depth * depth


def _terminal_score(board: Board, ply: int) -> float:
    """Score a terminal node; prefers shorter mates via ply offset."""
    current = board._turn_color
    opp = current.opponent
    my_tok = board._count_tokens(current)
    op_tok = board._count_tokens(opp)
    if my_tok == 0 and op_tok == 0:
        return 0.0
    if my_tok == 0:
        return -MATE + ply
    if op_tok == 0:
        return MATE - ply
    return 0.0


class Searcher:
    """Iterative-deepening negamax with alpha-beta, TT, killers, and history."""

    def __init__(self) -> None:
        self.tt = TranspositionTable()
        self.killers = [[None, None] for _ in range(MAX_PLY)]
        self.history: dict[tuple, int] = {}
        self.nodes = 0
        self.q_nodes = 0
        self.deadline = 0.0
        self.tt_hits = 0
        self.tt_lookups = 0
        self._path_hashes: set[int] = set()  # hashes of positions on the current search path

    def _reset(self) -> None:
        """Prepare for a new search: age TT, halve history, clear counters."""
        self.tt.soft_clear()
        self.killers = [[None, None] for _ in range(MAX_PLY)]
        self.history = {k: v >> 1 for k, v in self.history.items() if v > 1}
        self.nodes = 0
        self.q_nodes = 0
        self.tt_hits = 0
        self.tt_lookups = 0
        self._path_hashes = set()

    def search(self, board: Board, color: PlayerColor,
               time_budget: float) -> tuple[float, Action | None]:
        """Iterative deepening entry point; returns (score, best_action)."""
        self.deadline = time.monotonic() + time_budget
        self._reset()

        actions = legal_play_actions(board, board._turn_color)
        if not actions:
            return 0.0, None
        if len(actions) == 1:
            return 0.0, actions[0]

        best_move: Action = actions[0]
        best_score: float = 0.0
        t_start = time.monotonic()

        for depth in range(1, MAX_PLY + 1):
            iter_start = time.monotonic()
            try:
                score, move = self._root_search(board, color, depth, best_move)
            except SearchAborted:
                break

            best_score = score
            best_move = move

            iter_elapsed = time.monotonic() - iter_start
            total_nodes = self.nodes + self.q_nodes
            nps = total_nodes / max(iter_elapsed, 1e-9)
            tt_rate = self.tt_hits / max(self.tt_lookups, 1)
            print(
                f"[d={depth:2d}] score={score:+8.1f}  "
                f"nodes={self.nodes:>8,}  qnodes={self.q_nodes:>7,}  "
                f"nps={nps:>10,.0f}  TT={tt_rate:5.1%}  "
                f"move={move}"
            )

            if abs(best_score) > MATE - 1000:
                break # found a forced win/loss; no point searching deeper

            elapsed = time.monotonic() - t_start
            remaining = time_budget - elapsed
            if iter_elapsed > remaining * 0.25:
                break # next depth would likely exceed budget

        return best_score, best_move

    def _root_search(self, board: Board, color: PlayerColor,
                     depth: int, prev_best: Action) -> tuple[float, Action]:
        """Single-depth root search with PVS; prev_best searched first."""
        alpha = float(-INF)
        beta = float(INF)

        actions = legal_play_actions(board, board._turn_color)
        _order_moves(board, actions, prev_best, self.killers[0], self.history)

        best_score: float = float(-INF)
        best_move: Action = actions[0]
        root_hash = self.tt.get_hash(board)

        for i, action in enumerate(actions):
            self._check_time()
            board.apply_action(action)
            self._path_hashes.add(root_hash)
            try:
                if i == 0:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, ply=1)
                else:
                    score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, ply=1)
                    if alpha < score < beta:
                        score = -self._negamax(board, depth - 1, -beta, -score, ply=1)
            finally:
                self._path_hashes.discard(root_hash)
                board.undo_action()

            if score > best_score:
                best_score = score
                best_move = action
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        self.tt.store(root_hash, depth, best_score, TT_EXACT, best_move)
        return best_score, best_move

    def _negamax(self, board: Board, depth: int, alpha: float, beta: float, ply: int) -> float:
        """Negamax with alpha-beta, TT, PVS, killers, history, and repetition detection."""
        self.nodes += 1
        if self.nodes & (NODE_CHECK - 1) == 0:
            self._check_time()

        if board.game_over:
            return _terminal_score(board, ply)

        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply)

        h = self.tt.get_hash(board)

        if h in self._path_hashes:
            return -10 # draw penalty to discourage cycles

        self.tt_lookups += 1
        tt_score, hash_move = self.tt.lookup(h, depth, alpha, beta)
        if tt_score is not None:
            self.tt_hits += 1
            return tt_score

        actions = legal_play_actions(board, board._turn_color)
        if not actions:
            return 0.0

        killers_at_ply = self.killers[ply] if 0 <= ply < MAX_PLY else [None, None]
        _order_moves(board, actions, hash_move, killers_at_ply, self.history)

        orig_alpha = alpha
        best_score = float(-INF)
        best_action: Action | None = None
        first = True

        self._path_hashes.add(h)
        try:
            for action in actions:
                board.apply_action(action)
                try:
                    if first:
                        score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
                        first = False
                    else:
                        score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                        if alpha < score < beta:
                            score = -self._negamax(board, depth - 1, -beta, -score, ply + 1)
                finally:
                    board.undo_action()

                if score > best_score:
                    best_score = score
                    best_action = action
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    _record_cutoff(action, depth, ply, self.killers, self.history)
                    break
        finally:
            self._path_hashes.discard(h)

        if best_score <= orig_alpha:
            flag = TT_UPPER
        elif best_score >= beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT
        self.tt.store(h, depth, best_score, flag, best_action)

        return best_score

    def _quiescence(self, board: Board, alpha: float, beta: float, ply: int) -> float:
        """Quiescence search over captures/cascades to avoid horizon effect."""
        self.q_nodes += 1
        if (self.nodes + self.q_nodes) & (NODE_CHECK - 1) == 0:
            self._check_time()

        if board.game_over:
            return _terminal_score(board, ply)

        current = board._turn_color
        stand_pat = evaluate(board, current)

        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        all_actions = legal_play_actions(board, current)
        if not all_actions:
            return stand_pat

        # Use token counts to determine endgame — avoids a redundant board scan
        my_stacks = board._count_tokens(current)
        opp_stacks = board._count_tokens(current.opponent)
        endgame = (my_stacks <= 1 or opp_stacks <= 2 or (my_stacks + opp_stacks) <= 4)

        noisy: list[Action] = []
        for a in all_actions:
            if isinstance(a, EatAction):
                noisy.append(a)
            elif isinstance(a, CascadeAction):
                if endgame or _cascade_threatens_eat(board, a, current.opponent):
                    noisy.append(a)

        if not noisy:
            return stand_pat

        noisy.sort(key=lambda a: _noisy_score(board, a, current), reverse=True)

        for action in noisy:
            board.apply_action(action)
            try:
                score = -self._quiescence(board, -beta, -alpha, ply + 1)
            finally:
                board.undo_action()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def _check_time(self) -> None:
        """Raise SearchAborted if the deadline has passed."""
        if time.monotonic() >= self.deadline:
            raise SearchAborted()


def _cascade_threat_sum(board: Board, action: CascadeAction, opp_color: PlayerColor) -> int:
    """Sum of enemy token heights along the cascade ray (0 if none hit)."""
    h = board[action.coord].height
    dr, dc = action.direction.r, action.direction.c
    r0, c0 = action.coord.r, action.coord.c
    total = 0
    for i in range(1, h + 2):
        nr, nc = r0 + dr * i, c0 + dc * i
        if not (0 <= nr <= 7 and 0 <= nc <= 7):
            break
        cell = board[_COORD_CACHE[(nr, nc)]]
        if not cell.is_empty and cell.color == opp_color:
            total += cell.height
    return total


def _cascade_threatens_eat(board: Board, action: CascadeAction, opp_color: PlayerColor) -> bool:
    """True if this cascade lands adjacent to an enemy stack it can immediately eat."""
    return _cascade_threat_sum(board, action, opp_color) > 0


def _noisy_score(board: Board, action: Action, current: PlayerColor) -> int:
    """Priority score for quiescence move ordering (higher = searched first)."""
    if isinstance(action, EatAction):
        victim = board[action.coord + action.direction]
        return 2_000_000 + victim.height * 100

    if isinstance(action, CascadeAction):
        threat = _cascade_threat_sum(board, action, current.opponent)
        return 1_000_000 + threat * 50 + board[action.coord].height * 10

    return 0


def best_move(board: Board, color: PlayerColor, depth: int = 6,
              time_limit: float = 5.0, tt: TranspositionTable = None,
              searcher: 'Searcher | None' = None) -> Action:
    """Create a Searcher if needed and return the best action."""
    if searcher is None:
        searcher = Searcher()
        if tt is not None:
            searcher.tt = tt

    _, action = searcher.search(board, color, time_limit)

    if action is None:
        actions = legal_play_actions(board, board._turn_color)
        if actions:
            return actions[0]
        raise RuntimeError("No legal actions available")

    return action