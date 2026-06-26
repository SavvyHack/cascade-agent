from __future__ import annotations
from referee.game import PlaceAction, Coord
from .state import State, BOARD_N, NCELLS, RED
from .moves import gen_placement_moves


# Hand-tuned tables (consistent with the eval module).
_CENTRE_PRIORITY = [Coord(3, 3), Coord(3, 4), Coord(4, 3), Coord(4, 4)]

# Friendly-cluster bonus indexed by friendly-cardinal-neighbour count.
# Peaks at 2: a stack with two friendly neighbours can be reached and reinforced
# from two directions, supporting big merged stacks.
_CLUSTER_BONUS = [0.0, 3.0, 4.0, 3.0, 1.5]

# Precomputed cardinal neighbours per flat cell index.
_neighbours: list[list[int]] = []
for r in range(BOARD_N):
    for c in range(BOARD_N):
        adj = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_N and 0 <= nc < BOARD_N:
                adj.append(nr * BOARD_N + nc)
        _neighbours.append(adj)


def _placement_heuristic(state: State, candidate: PlaceAction) -> float:
    """
    Score one placement candidate; higher is better.

    The strategy is "defensive cluster": build a tight, central formation of
    our own stacks so we can later merge them into a tall cascading stack,
    without the over-aggressive proximity-to-enemy bonus that hurt A2 when
    playing Blue.

    Components:
      - Banded centralisation (inner/middle/outer).
      - Edge and corner penalties.
      - Friendly cluster bonus, peaking at 2 friendly neighbours.
      - Empty-neighbour bonus (room to move and grow).
      - Mild penalty for being adjacent to enemy stacks (don't sit in
        their cascade lanes during placement).
    """
    me = state.turn
    r, c = candidate.coord.r, candidate.coord.c
    idx = r * BOARD_N + c
    board = state.board

    score = 0.0

    # Centralisation: prefer the inner 4x4, then a middle ring, then outer.
    inner = 2 <= r <= 5 and 2 <= c <= 5
    middle = 1 <= r <= 6 and 1 <= c <= 6
    if inner:
        score += 3.0
    elif middle:
        score += 1.0

    # Edge / corner penalties.
    on_edge = r == 0 or r == 7 or c == 0 or c == 7
    on_corner = (r in (0, 7)) and (c in (0, 7))
    if on_edge:
        score -= 2.5
    if on_corner:
        score -= 2.0

    # Friendly-cluster bonus: depends on how many cardinal neighbours are ours.
    friendly_count = 0
    enemy_adj_count = 0
    empty_count = 0
    for n_idx in _neighbours[idx]:
        n = board[n_idx]
        if n == 0:
            empty_count += 1
        elif (n > 0) == (me == RED):
            friendly_count += 1
        else:
            enemy_adj_count += 1

    if friendly_count < len(_CLUSTER_BONUS):
        score += _CLUSTER_BONUS[friendly_count]
    else:
        score += _CLUSTER_BONUS[-1]

    # Empty neighbours: future room to move and merge.
    score += empty_count * 0.25

    # Mild penalty for sitting directly next to enemy stacks (we don't
    # generate placements adjacent to enemies anyway -- this is for cells
    # whose neighbourhood is dominated by enemy material).
    score -= enemy_adj_count * 1.5

    return score


def select_placement(state: State) -> PlaceAction:
    """Choose the best legal placement using the defensive-cluster heuristic."""
    candidates = gen_placement_moves(state)
    if not candidates:
        raise RuntimeError("No legal placements available")

    # Very first move of the game (Red only): grab the centre directly.
    # The inner 4x4 are equally tied on positional value, so picking one
    # deterministically avoids wasted compute and gives us the optimal start.
    if state.placement_count == 0:
        legal_coords = {a.coord for a in candidates}
        for centre in _CENTRE_PRIORITY:
            if centre in legal_coords:
                return PlaceAction(centre)

    best_action = candidates[0]
    best_score = _placement_heuristic(state, best_action)
    for candidate in candidates[1:]:
        candidate_score = _placement_heuristic(state, candidate)
        if candidate_score > best_score:
            best_action = candidate
            best_score = candidate_score
    return best_action
