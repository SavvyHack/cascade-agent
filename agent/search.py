from __future__ import annotations
import time
from referee.game import Action, PlaceAction, MoveAction, EatAction, CascadeAction
from .state import State, RED, BOARD_N
from .moves import gen_all_moves, gen_play_moves
from .evaluation import evaluate


# Sentinel values for search bounds and game outcomes.
INFINITY = 10_000_000       # used as ±alpha-beta window initial bounds
MATE_SCORE = 1_000_000      # base score for a win; we subtract ply so faster wins score higher
MAX_PLY = 64                # upper bound on recursion depth (for killer-slot tables)

# Transposition-table entry flags.
TT_EXACT = 0    # stored score is exact for this position
TT_LOWER = 1    # stored score is a lower bound (search produced a beta cutoff)
TT_UPPER = 2    # stored score is an upper bound (search failed low)


class SearchAborted(Exception):
    """Raised internally to unwind the search when the time budget expires."""


class Searcher:
    """
    Iterative-deepening alpha-beta searcher with PVS, a transposition table,
    quiescence search, killer-move and history heuristics.

    One Searcher persists for the lifetime of the agent so that the TT and
    history tables can carry over between moves -- the next ID iteration
    benefits from move-ordering information gathered by the previous one.
    """

    def __init__(self) -> None:
        # Transposition table: hash -> (depth, score, bound_flag, best_move)
        self.transposition_table: dict[int, tuple[int, int, int, Action | None]] = {}
        # Two killer moves per ply: quiet moves that caused recent cutoffs.
        self.killer_moves: list[list[Action | None]] = [[None, None] for _ in range(MAX_PLY)]
        # History heuristic: (action-type, src_r, src_c, dr, dc) -> cumulative score.
        self.history_scores: dict[tuple, int] = {}
        # Statistics, reset per top-level search call.
        self.nodes_searched = 0
        self.q_nodes_searched = 0
        self.deadline_monotonic = 0.0
        self.aborted = False

    # ---------------------------------------------------------- top-level

    def search(self, state: State, time_budget_seconds: float) -> tuple[int, Action | None]:
        """
        Top-level entry point. Run iterative deepening within the given time
        budget and return (score, best_move) from the side-to-move's
        perspective.

        Returns the deepest fully-completed iteration's result. If even the
        depth-1 search cannot complete (very rare), we fall back to the
        first move provided by the generator.
        """
        self.deadline_monotonic = time.monotonic() + time_budget_seconds
        self.aborted = False
        self.nodes_searched = 0
        self.q_nodes_searched = 0

        # No moves available -> draw/terminal handled by caller.
        moves = gen_all_moves(state)
        if not moves:
            return 0, None
        # Trivial single-move position: no need to search.
        if len(moves) == 1:
            return 0, moves[0]

        best_move: Action = moves[0]
        best_score = 0

        # Placement search is bounded shallowly; the heuristic handles
        # placement now, but we leave this in case it ever delegates here.
        max_depth = 64 if not state.is_placement_phase else 4

        for depth in range(1, max_depth + 1):
            try:
                score, move = self._root_search(state, depth, best_move)
            except SearchAborted:
                # Out of time mid-iteration -- keep the previous iteration's result.
                break
            best_score, best_move = score, move
            # Forced mate found: no point searching deeper.
            if abs(best_score) > MATE_SCORE - 1000:
                break
            # If we've burned more than 60% of our budget already, the next
            # iteration is unlikely to finish; bail out rather than waste it.
            elapsed = time.monotonic() - (self.deadline_monotonic - time_budget_seconds)
            if elapsed > 0.6 * time_budget_seconds:
                break

        return best_score, best_move

    # ---------------------------------------------------------- root

    def _root_search(self, state: State, depth: int, previous_best: Action) -> tuple[int, Action]:
        """
        Alpha-beta search at the root, with full window. The previous best
        move (from the prior iteration) is tried first via move ordering.
        """
        alpha, beta = -INFINITY, INFINITY
        moves = gen_all_moves(state)
        self._order_moves(state, moves, depth=depth, hash_move=previous_best, ply=0)

        best_score = -INFINITY
        best_move = moves[0]

        for i, action in enumerate(moves):
            self._check_time()
            state.apply(action)
            try:
                if i == 0:
                    # Principal variation: full-window search.
                    score = -self._negamax(state, depth - 1, -beta, -alpha, ply=1)
                else:
                    # PVS: null-window search; re-search with full window only on fail-high.
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
                # Beta cutoff at root: stop searching siblings.
                break

        # Cache the root result as an exact value.
        self.transposition_table[state.hash] = (depth, best_score, TT_EXACT, best_move)
        return best_score, best_move

    # ---------------------------------------------------------- negamax

    def _negamax(self, state: State, depth: int, alpha: int, beta: int, ply: int) -> int:
        """
        Recursive negamax with alpha-beta + PVS. Scores are from the
        perspective of the side to move at this node.
        """
        self.nodes_searched += 1
        # Cheap periodic time check (every 4096 nodes) -- adds negligible overhead.
        if self.nodes_searched & 4095 == 0:
            self._check_time()

        # Terminal: forced win/loss/draw before reaching depth limit.
        winner = state.winner()
        if winner is not None:
            if winner == 0:
                return 0
            # Faster mates score higher (or losses score "less bad").
            return -MATE_SCORE + ply if winner != state.turn else MATE_SCORE - ply

        # Leaf: hand off to quiescence so we don't evaluate mid-tactic positions.
        if depth <= 0:
            return self._quiescence(state, alpha, beta, ply)

        # Threefold-repetition draws are handled explicitly because the TT
        # could otherwise mask them.
        if state.is_threefold():
            return 0

        original_alpha = alpha
        hash_move: Action | None = None

        # Transposition table probe: reuse a stored result if it's at least
        # as deep as the current search and its bound is compatible with
        # the current alpha/beta window.
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
            # No legal action: stalemate draw.
            return 0
        self._order_moves(state, moves, depth=depth, hash_move=hash_move, ply=ply)

        best_score = -INFINITY
        best_move: Action | None = None
        is_first_move = True

        for action in moves:
            state.apply(action)
            try:
                if is_first_move:
                    # Principal variation: full window.
                    score = -self._negamax(state, depth - 1, -beta, -alpha, ply + 1)
                    is_first_move = False
                else:
                    # PVS: probe with null window first.
                    score = -self._negamax(state, depth - 1, -alpha - 1, -alpha, ply + 1)
                    if alpha < score < beta:
                        # The probe suggested this might beat alpha; re-search exactly.
                        score = -self._negamax(state, depth - 1, -beta, -score, ply + 1)
            finally:
                state.undo()

            if score > best_score:
                best_score = score
                best_move = action
            if score > alpha:
                alpha = score
            if alpha >= beta:
                # Beta cutoff: remember this move for ordering future siblings.
                self._record_cutoff(action, depth, ply)
                break

        # Classify the result before storing it in the TT.
        if best_score <= original_alpha:
            bound_flag = TT_UPPER       # we never beat alpha -> upper bound only
        elif best_score >= beta:
            bound_flag = TT_LOWER       # cutoff occurred -> lower bound only
        else:
            bound_flag = TT_EXACT       # value lies strictly inside the window
        self.transposition_table[state.hash] = (depth, best_score, bound_flag, best_move)
        return best_score

    # ---------------------------------------------------------- quiescence

    def _quiescence(self, state: State, alpha: int, beta: int, ply: int) -> int:
        """
        Quiescence search: at the depth horizon, keep extending through
        capture (EAT) sequences until the position is "quiet", so the static
        evaluator is never called in the middle of a forcing exchange.

        Only EAT moves are considered -- CASCADE could be tactical but is
        harder to bound depth-wise, and we want quiescence to terminate.
        """
        self.q_nodes_searched += 1
        if (self.nodes_searched + self.q_nodes_searched) & 4095 == 0:
            self._check_time()

        # Terminal still possible (e.g. a chain of eats ends the game).
        winner = state.winner()
        if winner is not None:
            if winner == 0:
                return 0
            return -MATE_SCORE + ply if winner != state.turn else MATE_SCORE - ply

        # Stand-pat: the score we'd accept if we chose to make no further
        # captures. Multiply by state.turn to convert from RED-perspective
        # to side-to-move perspective.
        stand_pat_score = evaluate(state) * state.turn
        if stand_pat_score >= beta:
            return beta
        if stand_pat_score > alpha:
            alpha = stand_pat_score

        # No tactical extensions during placement -- the evaluator handles it.
        if state.is_placement_phase:
            return stand_pat_score

        # Generate capture moves only.
        eat_moves = [m for m in gen_play_moves(state) if isinstance(m, EatAction)]
        if not eat_moves:
            return stand_pat_score

        # MVV-style ordering: try eating the largest victim first.
        board = state.board
        BN = BOARD_N

        def _victim_height(eat: EatAction) -> int:
            sr, sc = eat.coord.r, eat.coord.c
            dr = eat.direction.r
            dc = eat.direction.c
            return abs(board[(sr + dr) * BN + (sc + dc)])

        eat_moves.sort(key=_victim_height, reverse=True)

        for action in eat_moves:
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
        """
        Sort the move list in place by an ordering score (higher first).

        Priority:
          1. Hash move from the TT (proven good previously)
          2. EAT actions, ordered by victim height (MVV)
          3. CASCADE actions with high enemy push-threat
          4. MOVE merges with friendly stacks
          5. Killer-move bonuses for quiet MOVEs
          6. History-heuristic scores for everything else
        """
        board = state.board
        BN = BOARD_N
        # Local killer slots for this ply.
        killers = self.killer_moves[ply] if 0 <= ply < MAX_PLY else (None, None)
        killer_1, killer_2 = killers[0], killers[1]
        history_scores = self.history_scores

        def order_score(action: Action) -> int:
            # Hash move: try first by a large margin.
            if action == hash_move:
                return 10_000_000

            # EAT: most-valuable-victim sorting within the capture tier.
            if isinstance(action, EatAction):
                sr, sc = action.coord.r, action.coord.c
                dr = action.direction.r
                dc = action.direction.c
                victim_height = abs(board[(sr + dr) * BN + (sc + dc)])
                return 1_000_000 + victim_height * 100

            # CASCADE: estimate push-threat (sum of enemy heights in the path).
            if isinstance(action, CascadeAction):
                sr, sc = action.coord.r, action.coord.c
                dr = action.direction.r
                dc = action.direction.c
                stack_height = abs(board[sr * BN + sc])
                me_sign = state.turn
                push_threat = 0
                # Walk up to height+1 cells along the cascade direction.
                for i in range(1, stack_height + 2):
                    nr, nc = sr + dr * i, sc + dc * i
                    if not (0 <= nr < BN and 0 <= nc < BN):
                        break
                    cell = board[nr * BN + nc]
                    if cell != 0 and (cell > 0) != (me_sign == RED):
                        push_threat += abs(cell)
                return 500_000 + push_threat * 50

            # MOVE: merges bumped; killers and history add bonuses.
            if isinstance(action, MoveAction):
                sr, sc = action.coord.r, action.coord.c
                dr = action.direction.r
                dc = action.direction.c
                dest_cell = board[(sr + dr) * BN + (sc + dc)]
                me_sign = state.turn
                is_friendly_merge = dest_cell != 0 and ((dest_cell > 0) == (me_sign == RED))
                base_score = 200_000 if is_friendly_merge else 0
                # Killer bonuses (only one applies; second slot weights less).
                if action == killer_1:
                    base_score += 90_000
                elif action == killer_2:
                    base_score += 80_000
                base_score += history_scores.get(_history_key(action), 0)
                return base_score

            # PLACE: no useful ordering during play phase (shouldn't appear here).
            if isinstance(action, PlaceAction):
                return 0
            return 0

        moves.sort(key=order_score, reverse=True)

    # ---------------------------------------------------------- heuristics state

    def _record_cutoff(self, action: Action, depth: int, ply: int) -> None:
        """
        After a beta cutoff, record the move into killer slots and the
        history table so it's tried earlier in sibling positions.

        We restrict this to MOVE actions: captures already get strong MVV
        ordering, and cascades have their own ordering term.
        """
        if isinstance(action, MoveAction):
            # Killer slots: shift slot 0 down to slot 1 unless it's already this move.
            slots = self.killer_moves[ply] if 0 <= ply < MAX_PLY else None
            if slots is not None and slots[0] != action:
                slots[1] = slots[0]
                slots[0] = action
            # History weighted by depth squared so deep cutoffs count more.
            key = _history_key(action)
            self.history_scores[key] = self.history_scores.get(key, 0) + depth * depth

    # ---------------------------------------------------------- time guard

    def _check_time(self) -> None:
        """Raise SearchAborted if we've hit the search deadline."""
        if time.monotonic() >= self.deadline_monotonic:
            self.aborted = True
            raise SearchAborted()


def _history_key(action: Action) -> tuple:
    """
    Build a hashable key uniquely identifying an action's "signature"
    (kind + source cell + direction) for the history-heuristic table.
    """
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