from __future__ import annotations
import time
from typing import Callable
from referee.game import Action, PlaceAction, MoveAction, EatAction, CascadeAction, Direction
from .state import State, RED, BLUE, NCELLS, BOARD_N
from .moves import gen_all_moves, gen_play_moves, gen_placement_moves
from .evaluation import evaluate


INF = 10_000_000
MATE = 1_000_000
MAX_PLY = 64

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2


class SearchAborted(Exception):
    pass


class Searcher:

    def __init__(self) -> None:
        self.tt: dict[int, tuple[int, int, int, Action | None]] = {}
        self.killers: list[list[Action | None]] = [[None, None] for _ in range(MAX_PLY)]
        self.history: dict[tuple, int] = {}
        self.nodes = 0
        self.q_nodes = 0
        self.deadline = 0.0
        self.aborted = False

    def search(self, state: State, time_budget: float) -> tuple[int, Action | None]:
        self.deadline = time.monotonic() + time_budget
        self.aborted = False
        self.nodes = 0
        self.q_nodes = 0

        moves = gen_all_moves(state)
        if not moves:
            return 0, None
        if len(moves) == 1:
            return 0, moves[0]

        best_move: Action = moves[0]
        best_score = 0

        max_depth = 64 if not state.is_placement_phase else 4

        for depth in range(1, max_depth + 1):
            try:
                score, move = self._root_search(state, depth, best_move)
            except SearchAborted:
                break
            best_score, best_move = score, move
            if abs(best_score) > MATE - 1000:
                break
            if time.monotonic() - (self.deadline - time_budget) > 0.6 * time_budget:
                break

        return best_score, best_move

    def _root_search(self, state: State, depth: int, prev_best: Action) -> tuple[int, Action]:
        alpha, beta = -INF, INF
        moves = gen_all_moves(state)
        self._order_moves(state, moves, depth=depth, hash_move=prev_best, ply=0)

        best_score = -INF
        best_move = moves[0]

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

        self.tt[state.hash] = (depth, best_score, TT_EXACT, best_move)
        return best_score, best_move

    def _negamax(self, state: State, depth: int, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        if self.nodes & 4095 == 0:
            self._check_time()

        winner = state.winner()
        if winner is not None:
            if winner == 0:
                return 0
            return -MATE + ply if winner != state.turn else MATE - ply

        if depth <= 0:
            return self._quiescence(state, alpha, beta, ply)

        if state.is_threefold():
            return 0

        orig_alpha = alpha
        hash_move: Action | None = None

        tt_entry = self.tt.get(state.hash)
        if tt_entry is not None:
            tt_depth, tt_score, tt_flag, tt_move = tt_entry
            hash_move = tt_move
            if tt_depth >= depth:
                if tt_flag == TT_EXACT:
                    return tt_score
                if tt_flag == TT_LOWER and tt_score >= beta:
                    return tt_score
                if tt_flag == TT_UPPER and tt_score <= alpha:
                    return tt_score

        moves = gen_all_moves(state)
        if not moves:
            return 0
        self._order_moves(state, moves, depth=depth, hash_move=hash_move, ply=ply)

        best_score = -INF
        best_move: Action | None = None
        first = True

        for action in moves:
            state.apply(action)
            try:
                if first:
                    score = -self._negamax(state, depth - 1, -beta, -alpha, ply + 1)
                    first = False
                else:
                    score = -self._negamax(state, depth - 1, -alpha - 1, -alpha, ply + 1)
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

        if best_score <= orig_alpha:
            flag = TT_UPPER
        elif best_score >= beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT
        self.tt[state.hash] = (depth, best_score, flag, best_move)
        return best_score

    def _quiescence(self, state: State, alpha: int, beta: int, ply: int) -> int:
        self.q_nodes += 1
        if (self.nodes + self.q_nodes) & 4095 == 0:
            self._check_time()

        winner = state.winner()
        if winner is not None:
            if winner == 0:
                return 0
            return -MATE + ply if winner != state.turn else MATE - ply

        stand_pat = evaluate(state) * state.turn
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        if state.is_placement_phase:
            return stand_pat

        moves = [m for m in gen_play_moves(state) if isinstance(m, EatAction)]
        if not moves:
            return stand_pat

        board = state.board
        BN = BOARD_N

        def _victim_height(eat: EatAction) -> int:
            sr, sc = eat.coord.r, eat.coord.c
            dr = eat.direction.r
            dc = eat.direction.c
            return abs(board[(sr + dr) * BN + (sc + dc)])

        moves.sort(key=_victim_height, reverse=True)

        for action in moves:
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

    def _order_moves(self, state: State, moves: list[Action],
                     depth: int, hash_move: Action | None, ply: int) -> None:
        board = state.board
        BN = BOARD_N
        killers = self.killers[ply] if 0 <= ply < MAX_PLY else (None, None)
        k1, k2 = killers[0], killers[1]
        history = self.history

        def score(action: Action) -> int:
            if action == hash_move:
                return 10_000_000
            if isinstance(action, EatAction):
                sr, sc = action.coord.r, action.coord.c
                dr = action.direction.r
                dc = action.direction.c
                victim = abs(board[(sr + dr) * BN + (sc + dc)])
                return 1_000_000 + victim * 100
            if isinstance(action, CascadeAction):
                sr, sc = action.coord.r, action.coord.c
                dr = action.direction.r
                dc = action.direction.c
                h = abs(board[sr * BN + sc])
                me_sign = state.turn
                threat = 0
                for i in range(1, h + 2):
                    nr, nc = sr + dr * i, sc + dc * i
                    if not (0 <= nr < BN and 0 <= nc < BN):
                        break
                    v = board[nr * BN + nc]
                    if v != 0 and (v > 0) != (me_sign == RED):
                        threat += abs(v)
                return 500_000 + threat * 50
            if isinstance(action, MoveAction):
                sr, sc = action.coord.r, action.coord.c
                dr = action.direction.r
                dc = action.direction.c
                dest_val = board[(sr + dr) * BN + (sc + dc)]
                me_sign = state.turn
                is_merge = dest_val != 0 and ((dest_val > 0) == (me_sign == RED))
                base = 200_000 if is_merge else 0
                if action == k1:
                    base += 90_000
                elif action == k2:
                    base += 80_000
                base += history.get(_history_key(action), 0)
                return base
            if isinstance(action, PlaceAction):
                return 0
            return 0

        moves.sort(key=score, reverse=True)

    def _record_cutoff(self, action: Action, depth: int, ply: int) -> None:
        if isinstance(action, MoveAction):
            kls = self.killers[ply] if 0 <= ply < MAX_PLY else None
            if kls is not None:
                if kls[0] != action:
                    kls[1] = kls[0]
                    kls[0] = action
            key = _history_key(action)
            self.history[key] = self.history.get(key, 0) + depth * depth

    def _check_time(self) -> None:
        if time.monotonic() >= self.deadline:
            self.aborted = True
            raise SearchAborted()


def _history_key(action: Action) -> tuple:
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