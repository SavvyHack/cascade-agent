from __future__ import annotations
import time
from referee.game import Action, PlaceAction, MoveAction, EatAction, CascadeAction
from .state import State, RED, BOARD_N
from .moves import gen_all_moves, gen_play_moves
from .evaluation import evaluate


# ---------------------------------------------------------------- search constants

INFINITY = 10_000_000
MATE_SCORE = 1_000_000
MAX_PLY = 64

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

TT_MAX_ENTRIES = 1_000_000   # eviction kicks in here

ASPIRATION_DELTA = 50        # initial aspiration window half-width (centi-tokens)
ASPIRATION_FROM_DEPTH = 4    # only narrow the window from this depth onwards
IID_MIN_DEPTH = 4            # internal iterative deepening only above this depth
LMR_MIN_DEPTH = 3            # late-move reductions only at this depth or above
LMR_MIN_MOVE_INDEX = 3       # reduce only moves at index 3+ in the ordered list


class SearchAborted(Exception):
    """Unwinds the search when the time budget expires."""


class Searcher:
    """
    Iterative-deepening alpha-beta searcher with PVS, transposition table,
    quiescence (EAT + threatening CASCADE), killer-move and history
    heuristics, late-move reductions, internal iterative deepening, and
    aspiration windows. One Searcher persists for the whole game.
    """

    def __init__(self) -> None:
        # TT entry: (depth, score, flag, best_move).
        self.transposition_table: dict[int, tuple[int, int, int, Action | None]] = {}
        self.killer_moves: list[list[Action | None]] = [[None, None] for _ in range(MAX_PLY)]
        self.history_scores: dict[tuple, int] = {}
        # Hashes of positions on the current search path -- used to detect
        # in-search cycles and score them as draws.
        self._path_hashes: set[int] = set()

        # Statistics, reset per top-level search call.
        self.nodes_searched = 0
        self.q_nodes_searched = 0
        self.deadline_monotonic = 0.0
        self.aborted = False

    # ---------------------------------------------------------- top-level

    def search(self, state: State, time_budget_seconds: float) -> tuple[int, Action | None]:
        """
        Iterative deepening within ``time_budget_seconds``. Returns
        (score, best_move) from the side-to-move's perspective.
        """
        self.deadline_monotonic = time.monotonic() + time_budget_seconds
        self.aborted = False
        self.nodes_searched = 0
        self.q_nodes_searched = 0
        self._path_hashes = set()
        # Age the TT and decay history mildly between top-level searches.
        self._age_tables()

        moves = gen_all_moves(state)
        if not moves:
            return 0, None
        if len(moves) == 1:
            return 0, moves[0]

        best_move: Action = moves[0]
        best_score = 0
        start_time = time.monotonic()

        max_depth = 64 if not state.is_placement_phase else 4

        # Aspiration windows narrow the search around the previous score after
        # a stable iteration; if we fail high/low we widen and re-search.
        alpha = -INFINITY
        beta = INFINITY

        for depth in range(1, max_depth + 1):
            iteration_start = time.monotonic()

            # Use an aspiration window only once we've got a stable estimate.
            if depth >= ASPIRATION_FROM_DEPTH and abs(best_score) < MATE_SCORE - 1000:
                window = ASPIRATION_DELTA
                while True:
                    asp_alpha = best_score - window
                    asp_beta = best_score + window
                    try:
                        score, move = self._root_search(state, depth, best_move,
                                                        asp_alpha, asp_beta)
                    except SearchAborted:
                        return best_score, best_move
                    # Fail-low: widen below.
                    if score <= asp_alpha:
                        window *= 4
                        if window > 4000:
                            alpha = -INFINITY
                            try:
                                score, move = self._root_search(state, depth, best_move,
                                                                -INFINITY, asp_beta)
                            except SearchAborted:
                                return best_score, best_move
                            break
                        continue
                    # Fail-high: widen above.
                    if score >= asp_beta:
                        window *= 4
                        if window > 4000:
                            try:
                                score, move = self._root_search(state, depth, best_move,
                                                                asp_alpha, INFINITY)
                            except SearchAborted:
                                return best_score, best_move
                            break
                        continue
                    break
            else:
                try:
                    score, move = self._root_search(state, depth, best_move,
                                                    -INFINITY, INFINITY)
                except SearchAborted:
                    return best_score, best_move

            best_score = score
            best_move = move

            # Forced mate -- no point going deeper.
            if abs(best_score) > MATE_SCORE - 1000:
                break

            # If more than 60% of our budget is spent, the next depth is unlikely
            # to complete; bail out cleanly.
            elapsed = time.monotonic() - start_time
            if elapsed > 0.6 * time_budget_seconds:
                break

        return best_score, best_move

    # ---------------------------------------------------------- root

    def _root_search(self, state: State, depth: int, previous_best: Action,
                     alpha: int, beta: int) -> tuple[int, Action]:
        """Root alpha-beta with the prior iteration's best tried first."""
        moves = gen_all_moves(state)
        self._order_moves(state, moves, depth=depth, hash_move=previous_best, ply=0)

        best_score = -INFINITY
        best_move = moves[0]
        original_alpha = alpha
        root_hash = state.hash

        self._path_hashes.add(root_hash)
        try:
            for i, action in enumerate(moves):
                self._check_time()
                state.apply(action)
                try:
                    if i == 0:
                        score = -self._negamax(state, depth - 1, -beta, -alpha, ply=1)
                    else:
                        score = -self._negamax(state, depth - 1, -alpha - 1, -alpha, ply=1)
                        if alpha < score < beta:
                            score = -self._negamax(state, depth - 1, -beta, -score, ply=1)
                finally:
                    state.undo()

                if score > best_score:
                    best_score = score
                    best_move = action
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    break
        finally:
            self._path_hashes.discard(root_hash)

        # Cache the root result.
        if best_score <= original_alpha:
            flag = TT_UPPER
        elif best_score >= beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT
        self._tt_store(state.hash, depth, best_score, flag, best_move)
        return best_score, best_move

    # ---------------------------------------------------------- negamax

    def _negamax(self, state: State, depth: int, alpha: int, beta: int, ply: int) -> int:
        """
        Negamax with alpha-beta and PVS. Scores are from the side-to-move's
        perspective at this node.
        """
        self.nodes_searched += 1
        if self.nodes_searched & 4095 == 0:
            self._check_time()

        # Terminal first (forced win/loss/draw).
        winner = state.winner()
        if winner is not None:
            if winner == 0:
                return 0
            return -MATE_SCORE + ply if winner != state.turn else MATE_SCORE - ply

        # Leaf: quiescence.
        if depth <= 0:
            return self._quiescence(state, alpha, beta, ply)

        # In-search repetition: prefer draws to losses, but mostly avoid cycling.
        if state.hash in self._path_hashes:
            return 0
        # Real threefold (rare to hit here, but cheap to check).
        if state.is_threefold():
            return 0

        original_alpha = alpha
        hash_move: Action | None = None

        # Transposition probe.
        tt_entry = self.transposition_table.get(state.hash)
        if tt_entry is not None:
            stored_depth, stored_score, stored_flag, stored_move = tt_entry
            hash_move = stored_move
            if stored_depth >= depth:
                if stored_flag == TT_EXACT:
                    return stored_score
                if stored_flag == TT_LOWER and stored_score >= beta:
                    return stored_score
                if stored_flag == TT_UPPER and stored_score <= alpha:
                    return stored_score

        moves = gen_all_moves(state)
        if not moves:
            return 0   # stalemate draw

        # Internal iterative deepening: if no hash move and we're deep enough,
        # do a shallow search just to populate one (better ordering -> better cutoffs).
        if hash_move is None and depth >= IID_MIN_DEPTH:
            try:
                self._negamax(state, depth - 2, alpha, beta, ply)
            except SearchAborted:
                raise
            tt_entry = self.transposition_table.get(state.hash)
            if tt_entry is not None:
                hash_move = tt_entry[3]

        self._order_moves(state, moves, depth=depth, hash_move=hash_move, ply=ply)

        best_score = -INFINITY
        best_move: Action | None = None
        self._path_hashes.add(state.hash)
        try:
            for i, action in enumerate(moves):
                state.apply(action)
                try:
                    # Decide on LMR: only for late, non-tactical moves at
                    # sufficient depth. Captures, cascades, merges, and killers
                    # are *not* reduced.
                    is_tactical = isinstance(action, (EatAction, CascadeAction))
                    is_killer = False
                    if 0 <= ply < MAX_PLY:
                        ks = self.killer_moves[ply]
                        if action == ks[0] or action == ks[1]:
                            is_killer = True
                    reduce = (depth >= LMR_MIN_DEPTH and
                              i >= LMR_MIN_MOVE_INDEX and
                              not is_tactical and
                              not is_killer)

                    if i == 0:
                        # PV: full window, full depth.
                        score = -self._negamax(state, depth - 1, -beta, -alpha, ply + 1)
                    else:
                        # PVS null-window probe, with reduction for late-quiet moves.
                        search_depth = depth - 2 if reduce else depth - 1
                        score = -self._negamax(state, search_depth, -alpha - 1, -alpha, ply + 1)
                        # If reduced search beat alpha, re-search at full depth.
                        if reduce and score > alpha:
                            score = -self._negamax(state, depth - 1, -alpha - 1, -alpha, ply + 1)
                        # If null-window result lies inside (alpha, beta), full-window re-search.
                        if alpha < score < beta:
                            score = -self._negamax(state, depth - 1, -beta, -score, ply + 1)
                finally:
                    state.undo()

                if score > best_score:
                    best_score = score
                    best_move = action
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    self._record_cutoff(action, depth, ply)
                    break
        finally:
            self._path_hashes.discard(state.hash)

        if best_score <= original_alpha:
            flag = TT_UPPER
        elif best_score >= beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT
        self._tt_store(state.hash, depth, best_score, flag, best_move)
        return best_score

    # ---------------------------------------------------------- quiescence

    def _quiescence(self, state: State, alpha: int, beta: int, ply: int) -> int:
        """
        At the depth horizon, keep searching through EATs and *threatening*
        cascades so the static evaluator is never called in the middle of a
        forcing exchange.
        """
        self.q_nodes_searched += 1
        if (self.nodes_searched + self.q_nodes_searched) & 4095 == 0:
            self._check_time()

        winner = state.winner()
        if winner is not None:
            if winner == 0:
                return 0
            return -MATE_SCORE + ply if winner != state.turn else MATE_SCORE - ply

        # Stand-pat: the value if we choose to make no more captures.
        # Convert RED-perspective eval to side-to-move perspective.
        stand_pat = evaluate(state) * state.turn
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        # No tactical extensions during the placement phase.
        if state.is_placement_phase:
            return stand_pat

        # Build the noisy-move list: every EAT, plus cascades that hit an enemy.
        all_moves = gen_play_moves(state)
        if not all_moves:
            return stand_pat

        red_tokens = state._red_tokens
        blue_tokens = state._blue_tokens
        total = red_tokens + blue_tokens
        endgame = total <= 10

        noisy: list[Action] = []
        for action in all_moves:
            if isinstance(action, EatAction):
                noisy.append(action)
            elif isinstance(action, CascadeAction):
                # Include cascades that threaten enemy material, or any
                # cascade if we're in the endgame (every tactical shift counts).
                if endgame or _cascade_hits_enemy(state, action):
                    noisy.append(action)

        if not noisy:
            return stand_pat

        # MVV-style ordering: EATs first (largest victim), then high-threat cascades.
        board = state.board
        BN = BOARD_N
        me_sign = state.turn

        def noisy_score(action: Action) -> int:
            if isinstance(action, EatAction):
                sr, sc = action.coord.r, action.coord.c
                dr = action.direction.r
                dc = action.direction.c
                victim = abs(board[(sr + dr) * BN + (sc + dc)])
                return 2_000_000 + victim * 100
            # CASCADE
            return 1_000_000 + _cascade_threat_sum(state, action) * 50

        noisy.sort(key=noisy_score, reverse=True)

        for action in noisy:
            state.apply(action)
            try:
                score = -self._quiescence(state, -beta, -alpha, ply + 1)
            finally:
                state.undo()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    # ---------------------------------------------------------- move ordering

    def _order_moves(self, state: State, moves: list[Action],
                     depth: int, hash_move: Action | None, ply: int) -> None:
        """Sort the move list in place by ordering score (higher first)."""
        board = state.board
        BN = BOARD_N
        killers = self.killer_moves[ply] if 0 <= ply < MAX_PLY else (None, None)
        killer_1, killer_2 = killers[0], killers[1]
        history_scores = self.history_scores

        def order_score(action: Action) -> int:
            if action == hash_move:
                return 10_000_000

            if isinstance(action, EatAction):
                sr, sc = action.coord.r, action.coord.c
                dr = action.direction.r
                dc = action.direction.c
                victim_height = abs(board[(sr + dr) * BN + (sc + dc)])
                return 1_000_000 + victim_height * 100

            if isinstance(action, CascadeAction):
                sr, sc = action.coord.r, action.coord.c
                dr = action.direction.r
                dc = action.direction.c
                stack_height = abs(board[sr * BN + sc])
                me_sign = state.turn
                push_threat = 0
                for i in range(1, stack_height + 2):
                    nr, nc = sr + dr * i, sc + dc * i
                    if not (0 <= nr < BN and 0 <= nc < BN):
                        break
                    cell = board[nr * BN + nc]
                    if cell != 0 and (cell > 0) != (me_sign == RED):
                        push_threat += abs(cell)
                return 500_000 + push_threat * 50

            if isinstance(action, MoveAction):
                sr, sc = action.coord.r, action.coord.c
                dr = action.direction.r
                dc = action.direction.c
                dest_cell = board[(sr + dr) * BN + (sc + dc)]
                me_sign = state.turn
                is_friendly_merge = dest_cell != 0 and ((dest_cell > 0) == (me_sign == RED))
                base = 200_000 if is_friendly_merge else 0
                if action == killer_1:
                    base += 90_000
                elif action == killer_2:
                    base += 80_000
                base += history_scores.get(_history_key(action), 0)
                return base

            if isinstance(action, PlaceAction):
                return 0
            return 0

        moves.sort(key=order_score, reverse=True)

    # ---------------------------------------------------------- heuristics

    def _record_cutoff(self, action: Action, depth: int, ply: int) -> None:
        """On a beta cutoff, update killer slots and the history table."""
        if isinstance(action, MoveAction):
            slots = self.killer_moves[ply] if 0 <= ply < MAX_PLY else None
            if slots is not None and slots[0] != action:
                slots[1] = slots[0]
                slots[0] = action
            key = _history_key(action)
            self.history_scores[key] = self.history_scores.get(key, 0) + depth * depth

    # ---------------------------------------------------------- TT mgmt

    def _tt_store(self, hash_key: int, depth: int, score: int, flag: int,
                  best_move: Action | None) -> None:
        """Insert/replace; evict oldest insertion if over the cap."""
        existing = self.transposition_table.get(hash_key)
        if existing is not None and existing[0] > depth:
            return   # keep the deeper entry
        if len(self.transposition_table) >= TT_MAX_ENTRIES and hash_key not in self.transposition_table:
            # Evict the oldest entry by insertion order.
            self.transposition_table.pop(next(iter(self.transposition_table)))
        self.transposition_table[hash_key] = (depth, score, flag, best_move)

    def _age_tables(self) -> None:
        """
        Called once per top-level search. Halves history scores so older
        cutoffs decay; shallow-prunes the TT if it has grown beyond half cap.
        Killer slots are wiped per search since they're ply-indexed.
        """
        self.history_scores = {k: v >> 1 for k, v in self.history_scores.items() if v > 1}
        self.killer_moves = [[None, None] for _ in range(MAX_PLY)]
        if len(self.transposition_table) > TT_MAX_ENTRIES // 2:
            # Age depths down by 2; entries that fall below 0 are kept at 0
            # (they'll just not satisfy depth >= stored_depth checks).
            self.transposition_table = {
                h: (max(0, d - 2), s, f, m)
                for h, (d, s, f, m) in self.transposition_table.items()
            }

    # ---------------------------------------------------------- time guard

    def _check_time(self) -> None:
        if time.monotonic() >= self.deadline_monotonic:
            self.aborted = True
            raise SearchAborted()


# ---------------------------------------------------------------- helpers

def _history_key(action: Action) -> tuple:
    """Hashable key for the history-heuristic table."""
    if isinstance(action, MoveAction):
        return ('M', action.coord.r, action.coord.c,
                action.direction.r, action.direction.c)
    if isinstance(action, EatAction):
        return ('E', action.coord.r, action.coord.c,
                action.direction.r, action.direction.c)
    if isinstance(action, CascadeAction):
        return ('C', action.coord.r, action.coord.c,
                action.direction.r, action.direction.c)
    return ('P', action.coord.r, action.coord.c)


def _cascade_threat_sum(state: State, action: CascadeAction) -> int:
    """Sum of enemy heights along the cascade ray."""
    board = state.board
    BN = BOARD_N
    me_sign = state.turn
    sr, sc = action.coord.r, action.coord.c
    dr = action.direction.r
    dc = action.direction.c
    h = abs(board[sr * BN + sc])
    total = 0
    for i in range(1, h + 2):
        nr, nc = sr + dr * i, sc + dc * i
        if not (0 <= nr < BN and 0 <= nc < BN):
            break
        cell = board[nr * BN + nc]
        if cell != 0 and (cell > 0) != (me_sign == RED):
            total += abs(cell)
    return total


def _cascade_hits_enemy(state: State, action: CascadeAction) -> bool:
    """True if this cascade lands an enemy stack in its path."""
    return _cascade_threat_sum(state, action) > 0
