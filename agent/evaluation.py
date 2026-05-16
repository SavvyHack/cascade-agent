from __future__ import annotations
from .state import State, BOARD_N, NCELLS


TEMPO = 8                   # small initiative bonus to side-to-move;

# Feature weights, in units of 1/50 of a token (so 50 = one token of material).
# Material dominates; everything else is a tie-breaker among equal-material positions.
W_MATERIAL = 50
W_STACK_COUNT = 5
W_CAPTURE_THREAT = 25       # bonus when we can EAT an adjacent enemy
W_CAPTURE_RISK = 35         # asymmetric: defence weighted higher than offence
W_EDGE_VULNERABILITY = 6    # per (edge-score x stack-height)
W_CENTRE_BONUS = 1
              


# Manhattan distance from each cell to the nearest edge. Used for edge-penalty
# scoring: cells with edge_distance 0 are most cascade-vulnerable.
_edge_distance = []
for r in range(BOARD_N):
    for c in range(BOARD_N):
        _edge_distance.append(min(r, c, BOARD_N - 1 - r, BOARD_N - 1 - c))

# 1 if the cell sits in the central 4x4 (rows 2-5, cols 2-5), else 0.
_is_central = []
for r in range(BOARD_N):
    for c in range(BOARD_N):
        _is_central.append(1 if 2 <= r <= 5 and 2 <= c <= 5 else 0)

# For each cell, the list of its up-to-4 cardinal-neighbour cell indices.
# Precomputed once at module load so the evaluator does no adjacency math.
_neighbours: list[list[int]] = []
for r in range(BOARD_N):
    for c in range(BOARD_N):
        adj = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_N and 0 <= nc < BOARD_N:
                adj.append(nr * BOARD_N + nc)
        _neighbours.append(adj)


def evaluate(state: State) -> int:
    """
    Static evaluation of a non-terminal position, from RED's perspective.

    Returns a positive score if the position favours RED, negative if it
    favours BLUE. Score units are 1/100 of a token, so 100 corresponds
    roughly to a one-token material advantage. The search negates this as
    needed for the side to move.
    """
    board = state.board

    # First pass: count material, stacks, and central-square presence per side.
    red_tokens = 0
    blue_tokens = 0
    red_stack_count = 0
    blue_stack_count = 0
    red_tokens_in_centre = 0
    blue_tokens_in_centre = 0

    for idx in range(NCELLS):
        cell = board[idx]
        if cell == 0:
            continue
        if cell > 0:
            red_tokens += cell
            red_stack_count += 1
            if _is_central[idx]:
                red_tokens_in_centre += cell
        else:
            blue_tokens += -cell
            blue_stack_count += 1
            if _is_central[idx]:
                blue_tokens_in_centre += -cell

    # Combine the three position-summary terms.
    score = 0
    score += W_MATERIAL * (red_tokens - blue_tokens)
    score += W_STACK_COUNT * (red_stack_count - blue_stack_count)
    score += W_CENTRE_BONUS * (red_tokens_in_centre - blue_tokens_in_centre)

    # Second pass: capture-threat analysis. For every adjacent (own, enemy)
    # pair where our height >= theirs, we threaten an EAT next turn. Threats
    # and risks are tallied separately so they can carry different weights.
    red_eat_threat_value = 0
    blue_eat_threat_value = 0
    for idx in range(NCELLS):
        cell = board[idx]
        if cell == 0:
            continue
        our_height = cell if cell > 0 else -cell
        for n_idx in _neighbours[idx]:
            neighbour = board[n_idx]
            if neighbour == 0:
                continue
            # Skip if same colour (no threat between friendlies).
            if (cell > 0) == (neighbour > 0):
                continue
            target_height = neighbour if neighbour > 0 else -neighbour
            # We can EAT this neighbour if our height meets or exceeds theirs.
            if our_height >= target_height:
                if cell > 0:
                    red_eat_threat_value += target_height
                else:
                    blue_eat_threat_value += target_height
    score += W_CAPTURE_THREAT * red_eat_threat_value
    score -= W_CAPTURE_RISK * blue_eat_threat_value

    # Third pass: edge-vulnerability penalty. Stacks on or near the edge
    # can be pushed off the board by a hostile cascade; penalty scales
    # with stack height (bigger losses sting more).
    red_edge_penalty = 0
    blue_edge_penalty = 0
    for idx in range(NCELLS):
        cell = board[idx]
        if cell == 0:
            continue
        d = _edge_distance[idx]
        if d == 0:
            edge_factor = 3      # actual edge: heavy penalty
        elif d == 1:
            edge_factor = 1      # one cell off: lighter penalty
        else:
            edge_factor = 0      # safe interior
        if cell > 0:
            red_edge_penalty += edge_factor * cell
        else:
            blue_edge_penalty += edge_factor * (-cell)
    score -= W_EDGE_VULNERABILITY * red_edge_penalty
    score += W_EDGE_VULNERABILITY * blue_edge_penalty

    # Small initiative bonus to side-to-move; state.turn is
    # +1 for RED, -1 for BLUE, so this naturally flips sign.
    score += TEMPO * state.turn

    return score


def evaluate_for(state: State, color: int) -> int:
    """
    Static evaluation from the perspective of ``color`` (+1 RED, -1 BLUE).

    Convenience wrapper that flips the sign when called on behalf of BLUE,
    so the caller never needs to think about whose turn it is.
    """
    return color * evaluate(state)