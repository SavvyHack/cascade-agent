from __future__ import annotations
import time
from referee.game import PlayerColor, Action

from .state import State, RED, BLUE
from .moves import gen_all_moves
from .search import Searcher
from .placement import select_placement


# Total CPU budget per player per game (matches the project spec).
TOTAL_GAME_BUDGET_SECONDS = 180.0

# Safety margin reserved so we never spend the last of the budget.
TIME_SAFETY_MARGIN = 5.0

# Per-move clamps.
MIN_MOVE_TIME = 0.05
MAX_MOVE_TIME = 8.0


class Agent:
    """
    Game-playing agent (referee entry point).

    Internal State mirrors the referee's Board. Placement uses a positional
    heuristic; play uses iterative-deepening alpha-beta with PVS plus LMR,
    IID, aspiration windows, killer/history move ordering, transposition
    table, and a cascade-aware quiescence search.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        self._color: PlayerColor = color
        self._our_sign: int = RED if color == PlayerColor.RED else BLUE
        self._state: State = State()
        # One Searcher across the whole game so the TT, history and killers
        # carry over between moves -- ID benefits from prior orderings.
        self._searcher: Searcher = Searcher()
        # Track our own elapsed search time as a fallback when the referee
        # doesn't pass time_remaining.
        self._cpu_time_used: float = 0.0

    # ---------------------------------------------------------- referee API

    def action(self, **referee: dict) -> Action:
        """Called when it is our turn; returns the chosen Action."""
        state = self._state

        assert state.turn == self._our_sign, (
            f"Agent asked to act but internal state says it is "
            f"{'RED' if state.turn == RED else 'BLUE'}'s turn"
        )

        # Placement phase: heuristic, no search.
        if state.is_placement_phase:
            return select_placement(state)

        # Play phase: time-bounded iterative-deepening alpha-beta.
        time_budget = self._compute_time_budget(referee)
        search_start = time.monotonic()
        chosen_move: Action | None = None

        try:
            _, chosen_move = self._searcher.search(state, time_budget)
        except Exception as exc:
            # Search crashes shouldn't forfeit -- fall back to any legal move.
            print(f"[agent] search crashed: {exc!r}; falling back to first legal move")

        self._cpu_time_used += time.monotonic() - search_start

        if chosen_move is None:
            legal_moves = gen_all_moves(state)
            if legal_moves:
                chosen_move = legal_moves[0]
            else:
                raise RuntimeError("No legal actions available")

        return chosen_move

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """Called after any action; mirror it into our state."""
        expected_sign = RED if color == PlayerColor.RED else BLUE
        if self._state.turn != expected_sign:
            raise RuntimeError(
                f"Agent state desync: referee says {color} just acted, but "
                f"our turn is {'RED' if self._state.turn == RED else 'BLUE'}"
            )
        self._state.apply(action)

    # ---------------------------------------------------------- time mgmt

    def _compute_time_budget(self, referee: dict) -> float:
        """
        Decide seconds for this move.

        Take remaining CPU time, subtract a safety margin, divide by an estimate
        of remaining own-moves. Mid- and late-game positions get multipliers
        because they sharpen and reward deeper search. Clamp the result.
        """
        time_remaining = referee.get("time_remaining")
        if time_remaining is None:
            time_remaining = max(0.0, TOTAL_GAME_BUDGET_SECONDS - self._cpu_time_used)

        available_seconds = max(0.0, time_remaining - TIME_SAFETY_MARGIN)

        # We make roughly half of remaining turns; /3 is a reasonable estimate
        # accounting for games typically ending well before 300 turns.
        play_turn = self._state.play_turn_count
        estimated_remaining_moves = max(8, (300 - play_turn) // 3)

        budget_seconds = available_seconds / estimated_remaining_moves

        # Sharper positions mid- and late-game deserve more thinking time.
        if play_turn > 20:
            budget_seconds *= 1.4
        if play_turn > 60:
            budget_seconds *= 1.25
        if play_turn > 150:
            # Very late: positions are typically thin and tactical; spend more.
            budget_seconds *= 1.2

        return max(MIN_MOVE_TIME, min(MAX_MOVE_TIME, budget_seconds))
