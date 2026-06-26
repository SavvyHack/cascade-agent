from referee.game import PlayerColor, Action, GamePhase, Board, PlaceAction, Coord

from .actions import legal_place_actions, legal_play_actions
from .evaluate import _NEIGHBOURS, _COORD_TO_IDX
from .search import Searcher


class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
        self._color = color
        self._board = Board()
        self._turn_count = 0
        self._searcher = Searcher()

        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as RED (first player)")
            case PlayerColor.BLUE:
                print("Testing: I am playing as BLUE")

    def action(self, **referee: dict) -> Action:
        """Return chosen action for this turn."""
        time_left = referee.get('time_remaining', None)

        # Placement phase
        if self._board.phase == GamePhase.PLACEMENT:
            moves = legal_place_actions(self._board, self._color)
            opp_color = self._color.opponent
            is_first_mover = (self._board._placement_count == 0)

            # Very first placement: take the best available centre cell.
            # The four true centre cells of an 8x8 board are (3,3), (3,4),
            # (4,3), (4,4).  We prefer them in that order; the first one that
            # appears in the legal-move list (all should be free on an empty
            # board) is returned immediately, bypassing the general heuristic.
            if is_first_mover:
                _CENTRE_CELLS = [Coord(3, 3), Coord(3, 4), Coord(4, 3), Coord(4, 4)]
                legal_coords = {a.coord for a in moves}
                for cell in _CENTRE_CELLS:
                    if cell in legal_coords:
                        return PlaceAction(cell)

            def place_score(a) -> float:
                idx = _COORD_TO_IDX[a.coord]
                r, c = a.coord.r, a.coord.c
                score = 0.0

                # Prefer central control, avoid edges/corners
                inner = 2 <= r <= 5 and 2 <= c <= 5
                middle = 1 <= r <= 6 and 1 <= c <= 6
                on_edge = r == 0 or r == 7 or c == 0 or c == 7
                on_corner = (r in (0, 7)) and (c in (0, 7))

                score += 3.0 if inner else (1.0 if middle else 0.0)
                score -= 2.5 if on_edge else 0.0
                score -= 2.0 if on_corner else 0.0

                nbrs = _NEIGHBOURS[idx]
                friendly_nbrs = sum(1 for nbr in nbrs if self._board[nbr].color == self._color)
                empty_nbrs = sum(1 for nbr in nbrs if self._board[nbr].color is None)

                # Placing first: cluster aggressively in center
                if is_first_mover:
                    bonuses = [0.0, 3.0, 4.0, 3.0, 1.5]
                    score += bonuses[friendly_nbrs] if friendly_nbrs < len(bonuses) else 1.5
                    score += empty_nbrs * 0.2

                else:
                    # Placing second: balance clustering + pressure
                    bonuses = [0.0, 2.0, 2.5, 1.5, 0.5]
                    score += bonuses[friendly_nbrs] if friendly_nbrs < len(bonuses) else 0.5
                    score += empty_nbrs * 0.3

                    # Reward proximity setup to enemy
                    near_enemy = 0
                    for nbr in nbrs:
                        if self._board[nbr].is_empty:
                            for nbr2 in _NEIGHBOURS[_COORD_TO_IDX[nbr]]:
                                if self._board[nbr2].color == opp_color:
                                    near_enemy += 1
                                    break
                    score += near_enemy * 1.5

                return score

            return max(moves, key=place_score)

        # Play phase
        budget = self._compute_budget(time_left)
        _, move = self._searcher.search(self._board, self._color, budget)

        # Fallback if search fails
        if move is None:
            actions = legal_play_actions(self._board, self._board._turn_color)
            if actions:
                return actions[0]
            raise RuntimeError("No legal actions available")

        return move

    def _compute_budget(self, time_left) -> float:
        """Time allocation per move."""
        SAFETY = 5.0 # reserve buffer
        MIN_TIME = 0.05
        MAX_TIME = 8.0

        if time_left is None:
            return 5.0

        available = max(0.0, time_left - SAFETY)

        # Estimate remaining moves for this player
        turn = self._turn_count
        our_remaining = max(8, (300 - turn) // 4)

        budget = available / our_remaining

        # Increase thinking time in deeper phases
        if turn > 30:
            budget *= 1.3
        if turn > 80:
            budget *= 1.2

        return max(MIN_TIME, min(MAX_TIME, budget))

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """Update internal board state after each move."""
        if color == self._color:
            self._turn_count += 1
        self._board.apply_action(action)