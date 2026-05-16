# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

from __future__ import annotations
import math
import random
import time

from referee.game import PlayerColor, Coord, Action, PlaceAction, MoveAction, EatAction, CascadeAction
from .gameState import GameState, CARDINAL_DIRS, opponent, in_bounds, neighbour

# =============================================================================
# TRANSPOSITION TABLE FLAGS
# =============================================================================

EXACT = 0
LOWER = 1
UPPER = 2

_TTEntry = tuple[float, int, int]   # (value, depth_searched, flag)


# =============================================================================
# HELPERS
# =============================================================================

def manhattan(a: Coord, b: Coord) -> int:
    return abs(a.r - b.r) + abs(a.c - b.c)


def _action_key(a: Action) -> tuple:
    # Fix: tuple instead of repr() — much faster to compute and hash
    match a:
        case EatAction(coord, d):      return (0, coord.r, coord.c, d)
        case CascadeAction(coord, d):  return (1, coord.r, coord.c, d)
        case MoveAction(coord, d):     return (2, coord.r, coord.c, d)
        case PlaceAction(coord):       return (3, coord.r, coord.c)
        case _:                        return (4, repr(a))


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate(state: GameState, my_color: PlayerColor) -> float:
    terminated, winner = state.is_terminated()
    if terminated:
        if winner == my_color:  return  1_000_000.0
        elif winner is None:    return          0.0
        else:                   return -1_000_000.0

    enemy = opponent(my_color)

    my_tokens  = state.token_count(my_color)
    en_tokens  = state.token_count(enemy)
    my_stacks  = sum(1 for c, _ in state.board.values() if c == my_color)
    en_stacks  = sum(1 for c, _ in state.board.values() if c == enemy)
    my_cascade = sum(h for c, h in state.board.values() if c == my_color and h >= 2)
    en_cascade = sum(h for c, h in state.board.values() if c == enemy and h >= 2)

    # Immediate eat threats for both sides
    my_eat = 0
    en_eat = 0
    for coord, (color, height) in state.board.items():
        for d in CARDINAL_DIRS:
            nr, nc = neighbour(coord, d)
            if not in_bounds(nr, nc):
                continue
            dest = Coord(nr, nc)
            if dest not in state.board:
                continue
            dest_color, dest_height = state.board[dest]
            if color == my_color and dest_color == enemy and height >= dest_height:
                my_eat += 1
            elif color == enemy and dest_color == my_color and height >= dest_height:
                en_eat += 1

    # Proximity: close the gap if we can threaten; stay away if we're weaker
    my_coords = [(coord, h) for coord, (c, h) in state.board.items() if c == my_color]
    en_coords = [(coord, h) for coord, (c, h) in state.board.items() if c == enemy]

    proximity = 0.0
    if my_coords and en_coords:
        for en_coord, en_height in en_coords:
            threatening = [(coord, h) for coord, h in my_coords if h >= en_height]
            weak        = [(coord, h) for coord, h in my_coords if h <  en_height]
            if threatening:
                proximity -= min(manhattan(coord, en_coord) for coord, _ in threatening)
            elif weak:
                proximity += min(manhattan(coord, en_coord) for coord, _ in weak)

    return (
        3.0 * (my_tokens  - en_tokens)
      + 2.0 * (my_stacks  - en_stacks)
      + 1.5 * (my_cascade - en_cascade)
      + 2.5 * my_eat
      - 2.5 * en_eat
      + 0.2 * proximity
    )


# =============================================================================
# MOVE ORDERING
# =============================================================================

def order_actions(
    actions: list[Action],
    prev_scores: dict[tuple, float] | None = None,
) -> list[Action]:
    """
    Partition into EAT > CASCADE > MOVE/PLACE, then sort within each tier
    by descending score from the previous iteration (if provided).
    Uses a partition loop instead of sorted() — faster at shallow depths
    where we don't have prev_scores yet.
    """
    eats, cascades, rest = [], [], []
    for a in actions:
        if isinstance(a, EatAction):       eats.append(a)
        elif isinstance(a, CascadeAction): cascades.append(a)
        else:                              rest.append(a)

    if prev_scores:
        key = lambda a: -prev_scores.get(_action_key(a), 0.0)
        eats.sort(key=key)
        cascades.sort(key=key)
        rest.sort(key=key)

    return eats + cascades + rest


# =============================================================================
# PLACEMENT HEURISTIC
# =============================================================================

def smart_place(state: GameState, my_color: PlayerColor) -> PlaceAction:
    """
    Heuristic placement rewarding centrality, spread, and safety.
    default=4.0 on spread so the first placement (no existing friendly stacks)
    still differentiates positions rather than collapsing all spread scores to zero.
    """
    actions      = state.legal_actions(my_color)
    my_coords    = state.stack_coords(my_color)
    enemy_coords = state.stack_coords(opponent(my_color))
    center       = 3.5

    def score(action: PlaceAction) -> float:
        coord      = action.coord
        centrality = -((coord.r - center) ** 2 + (coord.c - center) ** 2)
        spread     = min((manhattan(coord, c) for c in my_coords), default=4.0)
        if enemy_coords:
            min_enemy = min(manhattan(coord, c) for c in enemy_coords)
            safety    = -max(0, 3 - min_enemy) * 2.0
        else:
            safety = 0.0
        return centrality * 0.5 + spread * 2.0 + safety

    return max(actions, key=score)


# =============================================================================
# MINIMAX WITH ALPHA-BETA PRUNING
# =============================================================================

def alphabeta(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    my_color: PlayerColor,
    start_time: float,
    time_limit: float,
    tt: dict[tuple, _TTEntry],
) -> float:
    if time.perf_counter() - start_time >= time_limit:
        raise TimeoutError

    # --- Transposition table lookup ---
    tt_key = (state.board_key(), maximizing)
    if tt_key in tt:
        tt_val, tt_depth, tt_flag = tt[tt_key]
        if tt_depth >= depth:
            if tt_flag == EXACT:             return tt_val
            if tt_flag == LOWER: alpha = max(alpha, tt_val)
            elif tt_flag == UPPER: beta = min(beta,  tt_val)
            if alpha >= beta:                return tt_val

    # --- Terminal / leaf ---
    terminated, winner = state.is_terminated()
    if terminated:
        val = (1_000_000.0 if winner == my_color else
               0.0         if winner is None      else
               -1_000_000.0)
        tt[tt_key] = (val, depth, EXACT)
        return val

    if depth == 0:
        val = evaluate(state, my_color)
        tt[tt_key] = (val, 0, EXACT)
        return val

    # --- Generate and cap actions per category ---
    raw = state.legal_actions()
    eats     = [a for a in raw if isinstance(a, EatAction)][:8]
    cascades = [a for a in raw if isinstance(a, CascadeAction)][:8]
    moves    = [a for a in raw if isinstance(a, (MoveAction, PlaceAction))][:16]
    actions  = eats + cascades + moves

    if not actions:
        tt[tt_key] = (0.0, depth, EXACT)
        return 0.0  # stalemate

    orig_alpha = alpha

    if maximizing:
        value = -math.inf
        for action in actions:
            child = state.apply_action(action)
            value = max(value, alphabeta(
                child, depth - 1, alpha, beta, False,
                my_color, start_time, time_limit, tt,
            ))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        flag = EXACT if orig_alpha < value < beta else (LOWER if value >= beta else UPPER)
    else:
        value = math.inf
        for action in actions:
            child = state.apply_action(action)
            value = min(value, alphabeta(
                child, depth - 1, alpha, beta, True,
                my_color, start_time, time_limit, tt,
            ))
            beta = min(beta, value)
            if alpha >= beta:
                break
        flag = EXACT if alpha < value < beta else (UPPER if value <= alpha else LOWER)

    tt[tt_key] = (value, depth, flag)
    return value


def best_action(
    state: GameState,
    my_color: PlayerColor,
    max_depth: int = 3,
    time_limit: float = 2.0,
    tt: dict | None = None,
) -> Action | None:
    """
    Iterative deepening alpha-beta.
    After each depth, per-action scores are recorded and passed to
    order_actions() for the next depth for real move-order reuse.
    Always returns the best move from the last fully-completed depth.
    """
    if tt is None:
        tt = {}

    raw_actions = state.legal_actions()
    if not raw_actions:
        return None

    start_time  = time.perf_counter()
    best_move   = raw_actions[0]
    prev_scores: dict[tuple, float] = {}

    for depth in range(1, max_depth + 1):
        try:
            best_val     = -math.inf
            current_best = best_move
            depth_scores: dict[tuple, float] = {}

            ordered = order_actions(raw_actions, prev_scores)

            for action in ordered:
                child = state.apply_action(action)
                val   = alphabeta(
                    child, depth - 1, -math.inf, math.inf,
                    False, my_color, start_time, time_limit, tt,
                )
                depth_scores[_action_key(action)] = val
                if val > best_val:
                    best_val     = val
                    current_best = action

            best_move   = current_best
            prev_scores = depth_scores

        except TimeoutError:
            break   # return best_move from last fully-completed depth

    return best_move


# =============================================================================
# AGENT
# =============================================================================

class Agent:
    """
    Cascade agent.
      - Placement phase : fast heuristic (smart_place)
      - Play phase      : iterative-deepening alpha-beta, depth up to 3

    Transposition table is instance-level so two agents in the same
    process never share or pollute each other's search state.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        self._color = color
        self._state = GameState()
        self._tt: dict[tuple, _TTEntry] = {}

    def action(self, **referee: dict) -> Action:
        if self._state.is_placement_phase:
            return smart_place(self._state, self._color)

        self._tt.clear()  # bound memory between turns

        act = best_action(
            self._state,
            self._color,
            max_depth=8,
            time_limit=2.0,
            tt=self._tt,
        )

        if act is None:
            actions = self._state.legal_actions()
            if not actions:
                raise RuntimeError("No legal actions available")
            return random.choice(actions)

        return act

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        self._state = self._state.apply_action(action)