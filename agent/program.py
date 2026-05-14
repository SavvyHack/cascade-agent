from __future__ import annotations
import time
from referee.game import (
    PlayerColor, Coord, Direction, Action,
    PlaceAction, MoveAction, EatAction, CascadeAction,
)
from .state import State, RED, BLUE
from .moves import gen_all_moves
from .search import Searcher
from .placement import select_placement


TOTAL_BUDGET = 180.0
SAFETY_MARGIN = 5.0
MIN_MOVE_TIME = 0.05
MAX_MOVE_TIME = 8.0

class Agent:

    def __init__(self, color: PlayerColor, **referee: dict):
        self._color: PlayerColor = color
        self._me: int = RED if color == PlayerColor.RED else BLUE
        self._state: State = State()
        self._searcher: Searcher = Searcher()
        self._time_used: float = 0.0

    def action(self, **referee: dict) -> Action:
        state = self._state

        assert state.turn == self._me, (
            f"Agent asked to act but internal state says it is "
            f"{'RED' if state.turn == RED else 'BLUE'}'s turn"
        )

        if state.is_placement_phase:
            return select_placement(state)

        budget = self._compute_time_budget(referee)
        t0 = time.monotonic()
        move: Action | None = None
        try:
            _, move = self._searcher.search(state, budget)
        except Exception as exc:
            print(f"[agent] search crashed: {exc!r}; falling back to first legal move")
        elapsed = time.monotonic() - t0
        self._time_used += elapsed

        if move is None:
            moves = gen_all_moves(state)
            if moves:
                move = moves[0]
            else:
                raise RuntimeError("No legal actions available")
        return move

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        expected = RED if color == PlayerColor.RED else BLUE
        if self._state.turn != expected:
            raise RuntimeError(
                f"Agent state desync: referee says {color} just acted, but "
                f"our turn is {'RED' if self._state.turn == RED else 'BLUE'}"
            )
        self._state.apply(action)

    def _compute_time_budget(self, referee: dict) -> float:
        time_remaining = referee.get("time_remaining")
        if time_remaining is None:
            time_remaining = max(0.0, TOTAL_BUDGET - self._time_used)

        available = max(0.0, time_remaining - SAFETY_MARGIN)

        play_turn = self._state.play_turn_count
        our_remaining = max(8, (300 - play_turn) // 4)

        budget = available / our_remaining
        if play_turn > 30:
            budget *= 1.3
        if play_turn > 80:
            budget *= 1.2

        return max(MIN_MOVE_TIME, min(MAX_MOVE_TIME, budget))