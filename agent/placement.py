from __future__ import annotations
from referee.game import PlaceAction
from .state import State, BOARD_N, NCELLS, RED
from .moves import gen_placement_moves


def _heuristic_score(state: State, action: PlaceAction) -> int:
    """
    Score a candidate placement; higher is better.

    The criteria, in rough order of importance:
      - Penalise edge and near-edge cells (cascade-vulnerable from move 1).
      - Reward central rows/columns (more options later).
      - Penalise placing adjacent to a friendly stack (a single hostile
        cascade could threaten both at once).
      - Reward placing at supporting distance (two cells away) from friendlies.
      - Penalise proximity to enemy stacks, scaled by their height.
    """
    place_r, place_c = action.coord.r, action.coord.c
    board = state.board
    me = state.turn

    score = 0

    # ---------- Centralisation: prefer the middle four rows and columns. ----
    # The distance from a "centre band" (rows 3 and 4 or cols 3 and 4) is
    # used as a soft penalty. Cells at r=3 or r=4 contribute 1 each, cells
    # at r=0 or r=7 contribute up to 7.
    centre_distance_r = abs(place_r - 3) + abs(place_r - 4)
    centre_distance_c = abs(place_c - 3) + abs(place_c - 4)
    score -= (centre_distance_r + centre_distance_c) * 3

    # ---------- Edge avoidance: heavy penalty for the actual edge. ----------
    # Edge stacks are immediately at risk of being pushed off the board.
    edge_distance = min(place_r, place_c, BOARD_N - 1 - place_r, BOARD_N - 1 - place_c)
    if edge_distance == 0:
        score -= 30          # on the edge: very bad
    elif edge_distance == 1:
        score -= 10          # one cell off the edge: still vulnerable

    # ---------- Relationship to existing pieces. ---------------------------
    # In a single board pass, find:
    #   - the Manhattan distance to our nearest friendly stack (for spacing), and
    #   - the cumulative "pressure" from nearby enemy stacks, weighted by their height.
    nearest_friend_distance = 99
    enemy_pressure = 0

    for idx in range(NCELLS):
        cell = board[idx]
        if cell == 0:
            continue

        other_r, other_c = idx // BOARD_N, idx % BOARD_N
        manhattan_distance = abs(other_r - place_r) + abs(other_c - place_c)
        is_friendly = (cell > 0) == (me == RED)

        if is_friendly:
            if manhattan_distance < nearest_friend_distance:
                nearest_friend_distance = manhattan_distance
        else:
            # Closer enemies project more pressure; their height matters too.
            enemy_height = cell if cell > 0 else -cell
            if manhattan_distance <= 2:
                enemy_pressure += enemy_height * (3 - manhattan_distance)
            elif manhattan_distance <= 4:
                enemy_pressure += enemy_height

    # Penalty/bonus based on spacing from our own stacks.
    if nearest_friend_distance < 99:
        if nearest_friend_distance == 1:
            # Adjacent to a friendly: a single hostile cascade could
            # threaten both stacks at once. Discourage strongly.
            score -= 20
        elif nearest_friend_distance == 2:
            # Supporting distance: close enough to coordinate, far
            # enough that one cascade can't engulf both. Encourage.
            score += 5
        else:
            # Further out: still fine, with a small bonus that grows
            # slightly with distance but caps at 4.
            score += min(nearest_friend_distance, 4)

    score -= enemy_pressure * 2

    return score


def select_placement(state: State) -> PlaceAction:
    """
    Choose the best legal placement using the positional heuristic.

    Called only during the placement phase (the first 8 turns total). We
    skip search entirely here because there's no tactical exchange yet --
    placements set up the board for the play phase, and a heuristic over
    candidate cells is both faster and more meaningful than minimax at
    depth-zero material differences.
    """
    candidates = gen_placement_moves(state)
    if not candidates:
        # Shouldn't be reachable given the rules, but guard against it
        # rather than silently returning a malformed move.
        raise RuntimeError("No legal placements available")

    best_action = candidates[0]
    best_score = _heuristic_score(state, best_action)

    for candidate in candidates[1:]:
        candidate_score = _heuristic_score(state, candidate)
        if candidate_score > best_score:
            best_action = candidate
            best_score = candidate_score

    return best_action