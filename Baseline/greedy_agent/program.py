"""
Greedy 1-ply lookahead testing agent.

Each turn, considers every legal action, simulates it on a private board copy,
and picks the action that maximises (own_tokens - opponent_tokens) on the
resulting position. No deeper lookahead -- doesn't consider opponent
responses.

Approximates the middle-tier Gradescope opponent (3 marks).

Self-contained -- does not depend on the main `agent` package. Shares the
random_agent's move logic by re-implementing it here so each opponent package
is fully independent.
"""

from __future__ import annotations

import copy
import random

from referee.game import (
    PlayerColor, Coord, Direction, Action,
    PlaceAction, MoveAction, EatAction, CascadeAction,
)


BOARD_N = 8
NCELLS = 64
PLACEMENT_TURNS = 8
INITIAL_STACK_HEIGHT = 3

RED = 1
BLUE = -1

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_NAMES = [Direction.Up, Direction.Down, Direction.Left, Direction.Right]


_ADJACENT: list[list[tuple[int, int]]] = []
for r in range(BOARD_N):
    for c in range(BOARD_N):
        adj = []
        for di, (dr, dc) in enumerate(DIRS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_N and 0 <= nc < BOARD_N:
                adj.append((nr * BOARD_N + nc, di))
        _ADJACENT.append(adj)


_DIR_DRDC = {
    Direction.Up: (-1, 0),
    Direction.Down: (1, 0),
    Direction.Left: (0, -1),
    Direction.Right: (0, 1),
}


class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
        self._color = color
        self._me = RED if color == PlayerColor.RED else BLUE
        self._board: list[int] = [0] * NCELLS
        self._placement_count = 0
        self._rng = random.Random()

    def action(self, **referee: dict) -> Action:
        moves = _legal_actions(self._board, self._me, self._placement_count)

        # During placement, just pick a non-edge cell at random. The greedy
        # criterion of "maximise token diff" doesn't differentiate among
        # placements (token diff is the same after any PLACE), so we'd be
        # picking arbitrarily anyway. Avoid the literal edges to make this
        # a marginally-stronger baseline.
        if self._placement_count < PLACEMENT_TURNS:
            non_edge = [
                m for m in moves
                if 1 <= m.coord.r <= 6 and 1 <= m.coord.c <= 6
            ]
            return self._rng.choice(non_edge or moves)

        best_move = moves[0]
        best_score = -10**9
        for m in moves:
            # Simulate move on a fresh copy of the board.
            sim_board = self._board.copy()
            _apply(sim_board, m, self._me)
            score = _material_diff(sim_board, self._me)
            if score > best_score:
                best_score = score
                best_move = m
        return best_move

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        sign = RED if color == PlayerColor.RED else BLUE
        _apply(self._board, action, sign)
        if isinstance(action, PlaceAction):
            self._placement_count += 1


# --- Free functions (shared logic, kept as module-level for clarity) -------


def _material_diff(board: list[int], me: int) -> int:
    my = sum(v if v > 0 else 0 for v in board) if me == RED \
        else -sum(v if v < 0 else 0 for v in board)
    opp = -sum(v if v < 0 else 0 for v in board) if me == RED \
        else sum(v if v > 0 else 0 for v in board)
    return my - opp


def _legal_actions(board: list[int], me: int, placement_count: int) -> list[Action]:
    if placement_count < PLACEMENT_TURNS:
        return _gen_placements(board, me, placement_count == 0)
    return _gen_play_actions(board, me)


def _gen_placements(board: list[int], me: int, is_first: bool) -> list[PlaceAction]:
    opponent = -me
    out: list[PlaceAction] = []
    for idx in range(NCELLS):
        if board[idx] != 0:
            continue
        if not is_first:
            blocked = False
            for n_idx, _ in _ADJACENT[idx]:
                v = board[n_idx]
                if (v > 0 and opponent == RED) or (v < 0 and opponent == BLUE):
                    blocked = True
                    break
            if blocked:
                continue
        out.append(PlaceAction(Coord(idx // BOARD_N, idx % BOARD_N)))
    return out


def _gen_play_actions(board: list[int], me: int) -> list[Action]:
    out: list[Action] = []
    for idx in range(NCELLS):
        v = board[idx]
        if me == RED:
            if v <= 0:
                continue
            h = v
        else:
            if v >= 0:
                continue
            h = -v
        coord = Coord(idx // BOARD_N, idx % BOARD_N)
        for n_idx, di in _ADJACENT[idx]:
            nv = board[n_idx]
            dn = DIR_NAMES[di]
            if nv == 0:
                out.append(MoveAction(coord, dn))
            elif (nv > 0) == (me == RED):
                out.append(MoveAction(coord, dn))
            else:
                if h >= abs(nv):
                    out.append(EatAction(coord, dn))
        if h >= 2:
            for di, dn in enumerate(DIR_NAMES):
                out.append(CascadeAction(coord, dn))
    return out


def _apply(board: list[int], action: Action, sign: int) -> None:
    if isinstance(action, PlaceAction):
        idx = action.coord.r * BOARD_N + action.coord.c
        board[idx] = sign * INITIAL_STACK_HEIGHT
        return

    if isinstance(action, MoveAction):
        dr, dc = _DIR_DRDC[action.direction]
        si = action.coord.r * BOARD_N + action.coord.c
        di = (action.coord.r + dr) * BOARD_N + (action.coord.c + dc)
        sv = board[si]
        dv = board[di]
        board[si] = 0
        board[di] = sv + dv if dv != 0 else sv
        return

    if isinstance(action, EatAction):
        dr, dc = _DIR_DRDC[action.direction]
        si = action.coord.r * BOARD_N + action.coord.c
        di = (action.coord.r + dr) * BOARD_N + (action.coord.c + dc)
        sv = board[si]
        board[si] = 0
        board[di] = sv
        return

    if isinstance(action, CascadeAction):
        _apply_cascade(board, action, sign)
        return

    raise ValueError(f"Unknown action {action!r}")


def _apply_cascade(board: list[int], action: CascadeAction, sign: int) -> None:
    src = action.coord
    si = src.r * BOARD_N + src.c
    h = abs(board[si])
    dr, dc = _DIR_DRDC[action.direction]

    working = dict(enumerate(board))
    working[si] = 0

    for i in range(1, h + 1):
        tr = src.r + dr * i
        tc = src.c + dc * i
        if not (0 <= tr < BOARD_N and 0 <= tc < BOARD_N):
            continue
        ti = tr * BOARD_N + tc
        if working[ti] != 0:
            _push_chain(working, ti, dr, dc)
        working[ti] = sign

    for idx, val in working.items():
        board[idx] = val


def _push_chain(working: dict, idx: int, dr: int, dc: int) -> None:
    cell = working[idx]
    if cell == 0:
        return
    r = idx // BOARD_N
    c = idx % BOARD_N
    nr = r + dr
    nc = c + dc
    if not (0 <= nr < BOARD_N and 0 <= nc < BOARD_N):
        working[idx] = 0
        return
    dest = nr * BOARD_N + nc
    if working[dest] != 0:
        _push_chain(working, dest, dr, dc)
    working[dest] = cell
    working[idx] = 0