from __future__ import annotations
from typing import Iterator
from referee.game import Action, PlaceAction, MoveAction, EatAction, CascadeAction, Coord, Direction
from .state import State, BOARD_N, NCELLS, DIRS, DIR_NAMES, RED, BLUE


_ADJACENT: list[list[tuple[int, int]]] = []
for r in range(BOARD_N):
    for c in range(BOARD_N):
        adj = []
        for di, (dr, dc) in enumerate(DIRS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_N and 0 <= nc < BOARD_N:
                adj.append((nr * BOARD_N + nc, di))
        _ADJACENT.append(adj)


def idx_to_coord(idx: int) -> Coord:
    return Coord(idx // BOARD_N, idx % BOARD_N)


def gen_placement_moves(state: State) -> list[PlaceAction]:
    board = state.board
    opponent = -state.turn
    is_first_placement = state.placement_count == 0

    moves: list[PlaceAction] = []
    for idx in range(NCELLS):
        if board[idx] != 0:
            continue
        if not is_first_placement:
            forbidden = False
            for n_idx, _ in _ADJACENT[idx]:
                v = board[n_idx]
                if (v > 0 and opponent == RED) or (v < 0 and opponent == BLUE):
                    forbidden = True
                    break
            if forbidden:
                continue
        moves.append(PlaceAction(idx_to_coord(idx)))
    return moves


def gen_play_moves(state: State) -> list[Action]:
    board = state.board
    me = state.turn
    moves: list[Action] = []

    for idx in range(NCELLS):
        v = board[idx]
        if me == RED:
            if v <= 0:
                continue
            our_height = v
        else:
            if v >= 0:
                continue
            our_height = -v

        coord = idx_to_coord(idx)

        for n_idx, di in _ADJACENT[idx]:
            n_val = board[n_idx]
            dir_name = DIR_NAMES[di]
            if n_val == 0:
                moves.append(MoveAction(coord, dir_name))
            elif (n_val > 0) == (me == RED):
                moves.append(MoveAction(coord, dir_name))
            else:
                if our_height >= abs(n_val):
                    moves.append(EatAction(coord, dir_name))

        if our_height >= 2:
            for di, dir_name in enumerate(DIR_NAMES):
                moves.append(CascadeAction(coord, dir_name))

    return moves


def gen_all_moves(state: State) -> list[Action]:
    if state.is_placement_phase:
        return gen_placement_moves(state)
    return gen_play_moves(state)


def has_any_legal_action(state: State) -> bool:
    if state.is_placement_phase:
        board = state.board
        opponent = -state.turn
        first = state.placement_count == 0
        for idx in range(NCELLS):
            if board[idx] != 0:
                continue
            if first:
                return True
            ok = True
            for n_idx, _ in _ADJACENT[idx]:
                v = board[n_idx]
                if (v > 0 and opponent == RED) or (v < 0 and opponent == BLUE):
                    ok = False
                    break
            if ok:
                return True
        return False

    board = state.board
    me = state.turn
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
        for n_idx, _ in _ADJACENT[idx]:
            n_val = board[n_idx]
            if n_val == 0:
                return True
            if (n_val > 0) == (me == RED):
                return True
            if h >= abs(n_val):
                return True
        if h >= 2:
            return True
    return False