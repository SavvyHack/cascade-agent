"""
Random testing agent.

This agent maintains a minimal internal board mirror, generates the set of
legal actions for its colour each turn, and returns one chosen uniformly at
random. Self-contained -- does not depend on the main `agent` package.

Approximates the lowest-tier Gradescope opponent (5 marks).
"""

from __future__ import annotations

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

# Cardinal directions: (dr, dc) and the matching Direction enum
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_NAMES = [Direction.Up, Direction.Down, Direction.Left, Direction.Right]


# Adjacency table: for each cell index, list of (neighbour index, direction index)
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
        moves = self._legal_actions()
        return self._rng.choice(moves)

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        sign = RED if color == PlayerColor.RED else BLUE
        self._apply(action, sign)

    # --- helpers ---

    def _legal_actions(self) -> list[Action]:
        if self._placement_count < PLACEMENT_TURNS:
            return self._gen_placements()
        return self._gen_play_actions()

    def _gen_placements(self) -> list[PlaceAction]:
        first = self._placement_count == 0
        opponent = -self._me
        moves: list[PlaceAction] = []
        for idx in range(NCELLS):
            if self._board[idx] != 0:
                continue
            if not first:
                blocked = False
                for n_idx, _ in _ADJACENT[idx]:
                    v = self._board[n_idx]
                    if (v > 0 and opponent == RED) or (v < 0 and opponent == BLUE):
                        blocked = True
                        break
                if blocked:
                    continue
            moves.append(PlaceAction(Coord(idx // BOARD_N, idx % BOARD_N)))
        return moves

    def _gen_play_actions(self) -> list[Action]:
        moves: list[Action] = []
        for idx in range(NCELLS):
            v = self._board[idx]
            if self._me == RED:
                if v <= 0:
                    continue
                h = v
            else:
                if v >= 0:
                    continue
                h = -v
            coord = Coord(idx // BOARD_N, idx % BOARD_N)
            for n_idx, di in _ADJACENT[idx]:
                nv = self._board[n_idx]
                dir_name = DIR_NAMES[di]
                if nv == 0:
                    moves.append(MoveAction(coord, dir_name))
                elif (nv > 0) == (self._me == RED):
                    moves.append(MoveAction(coord, dir_name))
                else:
                    if h >= abs(nv):
                        moves.append(EatAction(coord, dir_name))
            if h >= 2:
                for di, dir_name in enumerate(DIR_NAMES):
                    moves.append(CascadeAction(coord, dir_name))
        return moves

    def _apply(self, action: Action, sign: int) -> None:
        """Mirror the referee's mutation on our internal board."""
        if isinstance(action, PlaceAction):
            idx = action.coord.r * BOARD_N + action.coord.c
            self._board[idx] = sign * INITIAL_STACK_HEIGHT
            self._placement_count += 1
            return

        if isinstance(action, MoveAction):
            dr, dc = _DIR_DRDC[action.direction]
            si = action.coord.r * BOARD_N + action.coord.c
            di = (action.coord.r + dr) * BOARD_N + (action.coord.c + dc)
            sv = self._board[si]
            dv = self._board[di]
            self._board[si] = 0
            self._board[di] = sv + dv if dv != 0 else sv
            return

        if isinstance(action, EatAction):
            dr, dc = _DIR_DRDC[action.direction]
            si = action.coord.r * BOARD_N + action.coord.c
            di = (action.coord.r + dr) * BOARD_N + (action.coord.c + dc)
            sv = self._board[si]
            self._board[si] = 0
            self._board[di] = sv
            return

        if isinstance(action, CascadeAction):
            self._apply_cascade(action, sign)
            return

        raise ValueError(f"Unknown action {action!r}")

    def _apply_cascade(self, action: CascadeAction, sign: int) -> None:
        src = action.coord
        si = src.r * BOARD_N + src.c
        h = abs(self._board[si])
        dr, dc = _DIR_DRDC[action.direction]

        # Working copy so we can simulate pushes step by step
        working = dict(enumerate(self._board))

        # Step 1: remove source stack
        working[si] = 0

        # Step 2: walk h cells in direction, placing a 1-token at each
        for i in range(1, h + 1):
            tr = src.r + dr * i
            tc = src.c + dc * i
            if not (0 <= tr < BOARD_N and 0 <= tc < BOARD_N):
                continue
            ti = tr * BOARD_N + tc
            if working[ti] != 0:
                self._push_chain(working, ti, dr, dc)
            working[ti] = sign  # height-1 token of cascader's colour

        # Commit
        for idx, val in working.items():
            self._board[idx] = val

    @staticmethod
    def _push_chain(working: dict, idx: int, dr: int, dc: int) -> None:
        cell = working[idx]
        if cell == 0:
            return
        r = idx // BOARD_N
        c = idx % BOARD_N
        nr = r + dr
        nc = c + dc
        if not (0 <= nr < BOARD_N and 0 <= nc < BOARD_N):
            # Pushed off the edge: eliminated.
            working[idx] = 0
            return
        dest = nr * BOARD_N + nc
        if working[dest] != 0:
            Agent._push_chain(working, dest, dr, dc)
        working[dest] = cell
        working[idx] = 0