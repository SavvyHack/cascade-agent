# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

from referee.game import Board, PlayerColor, Coord, Direction, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction, GamePhase
from .search import choose_action


class Agent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Cascade game events.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        self._color = color
        self._board = Board()
        self._max_depth = 3

    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object.
        """
        # search more shallowly in Placement phase because there are many possible Place actions
        if self._board.phase == GamePhase.PLACEMENT:
            depth = 2
        else:
            depth = self._max_depth

        return choose_action(
            board=self._board,
            my_color=self._color,
            max_depth=depth,
        )

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state.
        """
        self._board.apply_action(action)