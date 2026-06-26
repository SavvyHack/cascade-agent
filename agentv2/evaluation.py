from __future__ import annotations
from .state import State, BOARD_N, NCELLS


# All weights in units of 1/100 of a token: 100 ~= one token of material.
W_MATERIAL = 100
W_STACK_COUNT = 3
W_CAPTURE_THREAT = 25       # offence: enemy tokens we can EAT next turn
W_CAPTURE_RISK = 35         # defence weighted higher than offence
W_FORK = 30                 # per extra eat target beyond the first, per attacker
W_EDGE_VULNERABILITY = 6
W_CENTRE_BONUS = 1
W_TEMPO = 5
W_MOBILITY = 2              # per pseudo-legal move (count, not list build)
W_CONNECTIVITY = 4          # per friendly cardinal-neighbour pair
W_TALL_BONUS = 4            # per token in the tallest stack of each side


# ---------------------------------------------------------------- precomputed tables

_edge_distance = []
for r in range(BOARD_N):
    for c in range(BOARD_N):
        _edge_distance.append(min(r, c, BOARD_N - 1 - r, BOARD_N - 1 - c))

_is_central = []
for r in range(BOARD_N):
    for c in range(BOARD_N):
        _is_central.append(1 if 2 <= r <= 5 and 2 <= c <= 5 else 0)

_neighbours: list[list[int]] = []
for r in range(BOARD_N):
    for c in range(BOARD_N):
        adj = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_N and 0 <= nc < BOARD_N:
                adj.append(nr * BOARD_N + nc)
        _neighbours.append(adj)


# ---------------------------------------------------------------- evaluation

def evaluate(state: State) -> int:
    """
    Static evaluation from RED's perspective. Positive favours Red, negative
    favours Blue. Score units are 1/100 of a token.

    Combines:
      - material (dominant)
      - stack-count delta
      - centre presence (weighted by token count)
      - capture threats and risks, with a fork bonus for multi-target attackers
      - edge vulnerability (cascadable off the board)
      - mobility (cheap pseudo-legal action count)
      - connectivity (friendly cardinal pairings, supports later merges)
      - tallest-stack bonus (cascade firepower)
      - tempo (small bonus to side to move)

    The evaluator is tapered very lightly: in the late endgame the edge
    penalty grows because losing a single token matters more. Other terms
    are kept fixed for simplicity and stable search behaviour.
    """
    board = state.board

    # First pass: per-side material, stack count, central tokens, tallest stack.
    red_tokens = state._red_tokens
    blue_tokens = state._blue_tokens
    red_stack_count = 0
    blue_stack_count = 0
    red_central = 0
    blue_central = 0
    red_max_height = 0
    blue_max_height = 0

    for idx in range(NCELLS):
        cell = board[idx]
        if cell == 0:
            continue
        if cell > 0:
            red_stack_count += 1
            if _is_central[idx]:
                red_central += cell
            if cell > red_max_height:
                red_max_height = cell
        else:
            blue_stack_count += 1
            mag = -cell
            if _is_central[idx]:
                blue_central += mag
            if mag > blue_max_height:
                blue_max_height = mag

    score = 0
    score += W_MATERIAL * (red_tokens - blue_tokens)
    score += W_STACK_COUNT * (red_stack_count - blue_stack_count)
    score += W_CENTRE_BONUS * (red_central - blue_central)
    score += W_TALL_BONUS * (red_max_height - blue_max_height)

    # Second pass: threats, risks, forks, connectivity, mobility, edge penalty.
    red_eat_threat = 0
    blue_eat_threat = 0
    red_forks = 0
    blue_forks = 0
    red_connectivity = 0
    blue_connectivity = 0
    red_mobility = 0
    blue_mobility = 0
    red_edge = 0
    blue_edge = 0

    for idx in range(NCELLS):
        cell = board[idx]
        if cell == 0:
            continue
        is_red = cell > 0
        our_height = cell if is_red else -cell

        # Edge vulnerability scales with stack height (bigger losses sting more).
        d = _edge_distance[idx]
        edge_factor = 3 if d == 0 else (1 if d == 1 else 0)
        if is_red:
            red_edge += edge_factor * our_height
        else:
            blue_edge += edge_factor * our_height

        # Adjacency-based features: count own eat targets (forks), connectivity,
        # and mobility for this stack.
        eat_targets = 0
        mobility_count = 0
        connectivity_count = 0
        for n_idx in _neighbours[idx]:
            neighbour = board[n_idx]
            if neighbour == 0:
                mobility_count += 1                # MOVE relocate is legal
                continue
            if (neighbour > 0) == is_red:
                # Friendly neighbour: contributes to connectivity and a MOVE-merge.
                connectivity_count += 1
                mobility_count += 1
            else:
                # Enemy neighbour: EAT legal iff our height >= theirs.
                target_height = neighbour if neighbour > 0 else -neighbour
                if our_height >= target_height:
                    eat_targets += 1
                    mobility_count += 1
                    if is_red:
                        red_eat_threat += target_height
                    else:
                        blue_eat_threat += target_height

        # Cascades add up to 4 more pseudo-moves at height >= 2.
        if our_height >= 2:
            mobility_count += 4

        if is_red:
            red_mobility += mobility_count
            red_connectivity += connectivity_count
            if eat_targets > 1:
                red_forks += eat_targets - 1
        else:
            blue_mobility += mobility_count
            blue_connectivity += connectivity_count
            if eat_targets > 1:
                blue_forks += eat_targets - 1

    score += W_CAPTURE_THREAT * red_eat_threat
    score -= W_CAPTURE_RISK * blue_eat_threat
    score += W_FORK * (red_forks - blue_forks)
    score += W_CONNECTIVITY * (red_connectivity - blue_connectivity) // 2
    # (connectivity counts each friendly pair twice -- one from each end -- so
    # we halve to recover the true pair count.)
    score += W_MOBILITY * (red_mobility - blue_mobility)

    # Edge: lightly taper -- if total material is low (late endgame), each
    # vulnerable token matters more.
    total_tokens = red_tokens + blue_tokens
    edge_mult = 100
    if total_tokens <= 12:
        edge_mult = 150
    elif total_tokens <= 18:
        edge_mult = 120
    score -= (W_EDGE_VULNERABILITY * (red_edge - blue_edge) * edge_mult) // 100

    # Tempo (Red +, Blue -).
    score += W_TEMPO * state.turn

    return score


def evaluate_for(state: State, color: int) -> int:
    """Eval from the perspective of `color` (+1 Red, -1 Blue)."""
    return color * evaluate(state)
