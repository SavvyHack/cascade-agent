from __future__ import annotations
import time
from referee.game import PlayerColor, Action

from .state import State, RED, BLUE
from .moves import gen_all_moves
from .search import Searcher
from .placement import select_placement


# Total CPU budget allowed per player per game (matches the spec's 180s limit).
TOTAL_GAME_BUDGET_SECONDS = 180.0

# Reserve a few seconds so we never spend the very last of the budget; this
# protects against timing imprecision and gives the search a hard floor to
# back off from.
TIME_SAFETY_MARGIN = 5.0

# Clamps on the per-move budget. The lower bound keeps even very late-game
# moves usable; the upper bound prevents us from over-investing early on
# easy positions.
MIN_MOVE_TIME = 0.05
MAX_MOVE_TIME = 8.0


class Agent:
    """
    Game-playing agent (the entry point required by the referee).

    Maintains an internal State that mirrors the referee's Board exactly,
    and runs iterative-deepening alpha-beta search to choose play-phase
    actions. The placement phase is delegated to a positional heuristic.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        self._color: PlayerColor = color
        self._our_sign: int = RED if color == PlayerColor.RED else BLUE
        self._state: State = State()
        # One Searcher instance lives across the whole game so the
        # transposition table and history heuristic carry over from move
        # to move (a substantial speedup in iterative deepening).
        self._searcher: Searcher = Searcher()
        # Track our own elapsed search time for budgeting when the referee
        # doesn't provide time_remaining.
        self._cpu_time_used: float = 0.0

    def action(self, **referee: dict) -> Action:
        """
        Called by the referee when it is our turn.

        Returns the chosen Action. Placement-phase moves come from the
        heuristic; play-phase moves come from iterative-deepening alpha-beta
        search constrained by a computed time budget.
        """
        state = self._state

        # Sanity check: the referee shouldn't ask us to act on the wrong turn.
        assert state.turn == self._our_sign, (
            f"Agent asked to act but internal state says it is "
            f"{'RED' if state.turn == RED else 'BLUE'}'s turn"
        )

        # Placement phase: use the heuristic, no search.
        if state.is_placement_phase:
            return select_placement(state)

        # Play phase: time-bounded iterative-deepening alpha-beta search.
        # Wrap in try/except so any unexpected failure falls back to a
        # legal move rather than forfeiting the game.
        time_budget = self._compute_time_budget(referee)
        search_start = time.monotonic()
        chosen_move: Action | None = None

        try:
            _, chosen_move = self._searcher.search(state, time_budget)
        except Exception as exc:
            # Defensive fallback: surface the error but keep playing.
            print(f"[agent] search crashed: {exc!r}; falling back to first legal move")

        self._cpu_time_used += time.monotonic() - search_start

        # If search returned nothing (e.g. aborted before any iteration
        # completed), grab any legal move so we don't forfeit.
        if chosen_move is None:
            legal_moves = gen_all_moves(state)
            if legal_moves:
                chosen_move = legal_moves[0]
            else:
                raise RuntimeError("No legal actions available")

        return chosen_move

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Called by the referee after any action has been applied (ours or
        the opponent's). We mirror the action into our internal state.
        """
        expected_sign = RED if color == PlayerColor.RED else BLUE
        if self._state.turn != expected_sign:
            # Our model has drifted out of sync with the referee. Better to
            # crash visibly than to keep searching a wrong board.
            raise RuntimeError(
                f"Agent state desync: referee says {color} just acted, but "
                f"our turn is {'RED' if self._state.turn == RED else 'BLUE'}"
            )
        self._state.apply(action)

    def _compute_time_budget(self, referee: dict) -> float:
        """
        Decide how many seconds to spend on the current move.

        Strategy: take the remaining CPU time, subtract a safety margin,
        and divide by an estimate of remaining moves. Apply small
        multipliers in the mid- and late-game where positions sharpen
        and deeper search pays off. Clamp the result so we neither
        starve early moves nor over-invest late.
        """
        # Prefer the referee's reported remaining time when available;
        # otherwise estimate from our own tracking.
        time_remaining = referee.get("time_remaining")
        if time_remaining is None:
            time_remaining = max(0.0, TOTAL_GAME_BUDGET_SECONDS - self._cpu_time_used)

        available_seconds = max(0.0, time_remaining - TIME_SAFETY_MARGIN)

        # Estimate how many more moves we'll need to make. Conservative:
        # divide remaining play-phase turns by 4 (instead of 2) so we
        # never overshoot even if the game runs unusually long.
        play_turn = self._state.play_turn_count
        estimated_remaining_moves = max(8, (300 - play_turn) // 4)

        budget_seconds = available_seconds / estimated_remaining_moves

        # Sharper positions mid- and late-game warrant deeper searches.
        if play_turn > 30:
            budget_seconds *= 1.3
        if play_turn > 80:
            budget_seconds *= 1.2

        # Final clamp to keep budgets sensible at both extremes.
        return max(MIN_MOVE_TIME, min(MAX_MOVE_TIME, budget_seconds))