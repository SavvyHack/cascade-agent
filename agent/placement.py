from __future__ import annotations
from referee.game import PlaceAction
from .state import State, BOARD_N, NCELLS, RED
from .moves import gen_placement_moves


def _heuristic_score(state: State, action: PlaceAction) -> int:
    r, c = action.coord.r, action.coord.c
    board = state.board
    me = state.turn

    score = 0

    centre_r = abs(r - 3) + abs(r - 4)
    centre_c = abs(c - 3) + abs(c - 4)
    score -= (centre_r + centre_c) * 3

    edge_dist = min(r, c, BOARD_N - 1 - r, BOARD_N - 1 - c)
    if edge_dist == 0:
        score -= 30
    elif edge_dist == 1:
        score -= 10

    min_friend_dist = 99
    enemy_pressure = 0
    for idx in range(NCELLS):
        v = board[idx]
        if v == 0:
            continue
        or_r, or_c = idx // BOARD_N, idx % BOARD_N
        d = abs(or_r - r) + abs(or_c - c)
        is_mine = (v > 0) == (me == RED)
        if is_mine:
            if d < min_friend_dist:
                min_friend_dist = d
        else:
            h = v if v > 0 else -v
            if d <= 2:
                enemy_pressure += h * (3 - d)
            elif d <= 4:
                enemy_pressure += h

    if min_friend_dist < 99:
        if min_friend_dist == 1:
            score -= 20
        elif min_friend_dist == 2:
            score += 5
        else:
            score += min(min_friend_dist, 4)

    score -= enemy_pressure * 2

    return score


def select_placement(state: State) -> PlaceAction:
    moves = gen_placement_moves(state)
    if not moves:
        raise RuntimeError('No legal placements available')
    best = moves[0]
    best_score = _heuristic_score(state, best)
    for m in moves[1:]:
        s = _heuristic_score(state, m)
        if s > best_score:
            best = m
            best_score = s
    return best