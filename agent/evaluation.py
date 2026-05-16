from __future__ import annotations
from .state import State, BOARD_N, NCELLS, RED, BLUE


W_MATERIAL = 100
W_STACK = 5
W_CAPTURE_THREAT = 25
W_CAPTURE_RISK = 35
W_EDGE_VULN = 6
W_CENTRE = 1


_EDGE_DIST = []
for r in range(BOARD_N):
    for c in range(BOARD_N):
        _EDGE_DIST.append(min(r, c, BOARD_N - 1 - r, BOARD_N - 1 - c))

_CENTRAL = []
for r in range(BOARD_N):
    for c in range(BOARD_N):
        _CENTRAL.append(1 if 2 <= r <= 5 and 2 <= c <= 5 else 0)

_NEIGHBOURS: list[list[int]] = []
for r in range(BOARD_N):
    for c in range(BOARD_N):
        neigh = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_N and 0 <= nc < BOARD_N:
                neigh.append(nr * BOARD_N + nc)
        _NEIGHBOURS.append(neigh)


def evaluate(state: State) -> int:
    board = state.board

    red_material = 0
    blue_material = 0
    red_stacks = 0
    blue_stacks = 0
    red_centre = 0
    blue_centre = 0

    for idx in range(NCELLS):
        v = board[idx]
        if v == 0:
            continue
        if v > 0:
            red_material += v
            red_stacks += 1
            if _CENTRAL[idx]:
                red_centre += v
        else:
            blue_material += -v
            blue_stacks += 1
            if _CENTRAL[idx]:
                blue_centre += -v

    score = 0
    score += W_MATERIAL * (red_material - blue_material)
    score += W_STACK * (red_stacks - blue_stacks)
    score += W_CENTRE * (red_centre - blue_centre)

    red_threat = 0
    blue_threat = 0

    for idx in range(NCELLS):
        v = board[idx]
        if v == 0:
            continue
        my_h = v if v > 0 else -v
        for n_idx in _NEIGHBOURS[idx]:
            nv = board[n_idx]
            if nv == 0:
                continue
            if (v > 0) == (nv > 0):
                continue
            n_h = nv if nv > 0 else -nv
            if my_h >= n_h:
                if v > 0:
                    red_threat += n_h
                else:
                    blue_threat += n_h
    score += W_CAPTURE_THREAT * red_threat
    score -= W_CAPTURE_RISK * blue_threat

    red_edge_pen = 0
    blue_edge_pen = 0
    for idx in range(NCELLS):
        v = board[idx]
        if v == 0:
            continue
        d = _EDGE_DIST[idx]
        if d == 0:
            pen = 3
        elif d == 1:
            pen = 1
        else:
            pen = 0
        if v > 0:
            red_edge_pen += pen * v
        else:
            blue_edge_pen += pen * (-v)

    score -= W_EDGE_VULN * red_edge_pen
    score += W_EDGE_VULN * blue_edge_pen

    score += 8 * state.turn

    return score


def evaluate_for(state: State, color: int) -> int:
    return color * evaluate(state)