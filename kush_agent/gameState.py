from __future__ import annotations
from typing import Iterator
from referee.game import PlayerColor, Coord, Direction, Action, PlaceAction, MoveAction, EatAction, CascadeAction

BOARD_SIZE = 8
PLACEMENT_TURNS = 8       
PLAY_TURN_LIMIT = 300
INITIAL_STACK_HEIGHT = 3

DIR: dict[Direction, tuple[int, int]] = {
    Direction.Up:        (-1,  0),
    Direction.Down:      ( 1,  0),
    Direction.Left:      ( 0, -1),
    Direction.Right:     ( 0,  1),
    Direction.UpLeft:    (-1, -1),
    Direction.UpRight:   (-1,  1),
    Direction.DownLeft:  ( 1, -1),
    Direction.DownRight: ( 1,  1),
}

# All four directions as a constant tuple
CARDINAL_DIRS: tuple[Direction, ...] = (Direction.Up, Direction.Down, Direction.Left, Direction.Right)
Board = dict[Coord, tuple[PlayerColor, int]]

# Helper functions
def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

def neighbour(coord: Coord, direction: Direction, steps: int = 1) -> tuple[int, int]:
    dr, dc = DIR[direction]
    return coord.r + dr * steps, coord.c + dc * steps

def opponent(color: PlayerColor) -> PlayerColor:
    return PlayerColor.BLUE if color == PlayerColor.RED else PlayerColor.RED

class GameState:
    """
    A complete, self-contained representation of a Cascade game state.
    """

    def __init__(self) -> None:
        self.board: Board = {}
        self.current_color: PlayerColor = PlayerColor.RED  # Red always goes first
        self.placement_count: int = 0   
        self.play_turn_count: int = 0   
        self._position_history: dict[tuple, int] = {}

    @property
    def is_placement_phase(self) -> bool:
        return self.placement_count < PLACEMENT_TURNS

    @property
    def opponent(self) -> PlayerColor:
        return opponent(self.current_color)

    def token_count(self, color: PlayerColor) -> int:
        """Total number of individual tokens, not stacks, owned by color."""
        return sum(h for c, h in self.board.values() if c == color)

    def stack_coords(self, color: PlayerColor) -> list[Coord]:
        """All coordinates where color has a stack."""
        return [coord for coord, (c, _) in self.board.items() if c == color]

    # Managing legal actions
    def legal_actions(self, color: PlayerColor | None = None) -> list[Action]:
        if color is None:
            color = self.current_color
        if self.is_placement_phase:
            return list(self.legal_placements(color))
        return list(self.legal_play_actions(color))

    def legal_placements(self, color: PlayerColor) -> Iterator[PlaceAction]:
        opp = opponent(color)
        forbidden: set[Coord] = set()

        if self.placement_count > 0:
            for coord, (c, _) in self.board.items():
                if c == opp:
                    for d in CARDINAL_DIRS:
                        nr, nc = neighbour(coord, d)
                        if in_bounds(nr, nc):
                            forbidden.add(Coord(nr, nc))

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                coord = Coord(r, c)
                if coord not in self.board and coord not in forbidden:
                    yield PlaceAction(coord)

    def legal_play_actions(self, color: PlayerColor) -> Iterator[Action]:
        for coord, (c, height) in self.board.items():
            if c != color:
                continue

            for direction in CARDINAL_DIRS:
                nr, nc = neighbour(coord, direction)
                # CASCADE is always valid for height ≥ 2, regardless of what lies in that direction
                if height >= 2:
                    yield CascadeAction(coord, direction)
                if not in_bounds(nr, nc):
                    continue
                dest = Coord(nr, nc)

                if dest not in self.board:
                    # Empty cell → relocate
                    yield MoveAction(coord, direction)
                else:
                    dest_color, dest_height = self.board[dest]
                    if dest_color == color:
                        # Friendly → merge
                        yield MoveAction(coord, direction)
                    else:
                        # Enemy → eat, only if attacker height ≥ target height
                        if height >= dest_height:
                            yield EatAction(coord, direction)

    # Applying actions
    def apply_action(self, action: Action) -> GameState:
        """
        Return new GameState from applying `action`. Original state never modified.
        """
        new = self.copy()
        color = new.current_color
        match action:
            case PlaceAction(coord):
                new.board[coord] = (color, INITIAL_STACK_HEIGHT)
                new.placement_count += 1

            case MoveAction(coord, direction):
                _, height = new.board.pop(coord)
                nr, nc = neighbour(coord, direction)
                dest = Coord(nr, nc)
                if dest in new.board:
                    _, dest_height = new.board[dest]
                    new.board[dest] = (color, height + dest_height)
                else:
                    new.board[dest] = (color, height)

            case EatAction(coord, direction):
                _, height = new.board.pop(coord)
                nr, nc = neighbour(coord, direction)
                dest = Coord(nr, nc)
                new.board[dest] = (color, height)

            case CascadeAction(coord, direction):
                _, height = new.board.pop(coord)
                new.apply_cascade(coord, direction, height, color)

        # Advance turn to opponent
        new.current_color = opponent(color)
        # Track position history in play phase
        if not new.is_placement_phase:
            new.play_turn_count += 1
            key = new.board_key()
            new._position_history[key] = new._position_history.get(key, 0) + 1
        return new

    def apply_cascade(
        self,
        origin: Coord,
        direction: Direction,
        height: int,
        color: PlayerColor,
    ) -> None:
        """
        Spread `height` tokens from `origin` in `direction`, pushing any stacks encountered.
        """
        for i in range(1, height + 1):
            nr, nc = neighbour(origin, direction, i)
            if not in_bounds(nr, nc):
                # This and all remaining tokens are lost off the board
                break
            dest = Coord(nr, nc)
            if dest in self.board:
                self.push_stack(dest, direction)
            self.board[dest] = (color, 1)

    def push_stack(self, coord: Coord, direction: Direction) -> None:
        """
        Recursively push the stack at `coord` one step in `direction`. If pushed off the board, the stack is eliminated.
        """
        nr, nc = neighbour(coord, direction)
        if not in_bounds(nr, nc):
            del self.board[coord]
            return
        new_coord = Coord(nr, nc)
        if new_coord in self.board:
            # Recursively push the stack that's blocking first
            self.push_stack(new_coord, direction)
        self.board[new_coord] = self.board.pop(coord)

    # Termination state detection
    def is_terminated(self) -> tuple[bool, PlayerColor | None]:
        """
        Check whether the game has ended. Winner is None for draw or winning PlayerColor.
        """
        # 1) Elimination
        red_alive  = any(c == PlayerColor.RED  for c, _ in self.board.values())
        blue_alive = any(c == PlayerColor.BLUE for c, _ in self.board.values())
        if not red_alive:
            return True, PlayerColor.BLUE
        if not blue_alive:
            return True, PlayerColor.RED
        if self.is_placement_phase:
            return False, None

        # 2) Threefold repetition
        key = self.board_key()
        if self._position_history.get(key, 0) >= 3:
            return True, None  # draw

        # 3) Stalemate
        if next(self.legal_play_actions(self.current_color), None) is None:
            return True, None  # draw

        # 4) Turn limit
        if self.play_turn_count >= PLAY_TURN_LIMIT:
            red_tokens  = self.token_count(PlayerColor.RED)
            blue_tokens = self.token_count(PlayerColor.BLUE)
            if red_tokens > blue_tokens:
                return True, PlayerColor.RED
            elif blue_tokens > red_tokens:
                return True, PlayerColor.BLUE
            else:
                return True, None
        return False, None

    def board_key(self) -> tuple:
        """
        A hashable snapshot of the board & whose turn it is. Used for threefold-repetition detection.
        """
        return (
            self.current_color,
            tuple(sorted(
                (coord.r, coord.c, color.value, height)
                for coord, (color, height) in self.board.items()
            )),
        )

    def copy(self) -> GameState:
        """Copy scalars, board and position history."""
        new = object.__new__(GameState)
        new.board = dict(self.board)                       
        new.current_color = self.current_color
        new.placement_count = self.placement_count
        new.play_turn_count = self.play_turn_count
        new._position_history = dict(self._position_history)
        return new

    def __repr__(self) -> str:
        """ASCII board dump for debugging."""
        lines = [f"Turn: {'Placement' if self.is_placement_phase else 'Play'} | "
                 f"Current: {self.current_color} | " f"Play turns: {self.play_turn_count}"]
        for r in range(BOARD_SIZE):
            row = []
            for c in range(BOARD_SIZE):
                coord = Coord(r, c)
                if coord in self.board:
                    col, h = self.board[coord]
                    tag = "R" if col == PlayerColor.RED else "B"
                    row.append(f"{tag}{h}")
                else:
                    row.append("..")
            lines.append(" ".join(row))
        red_tok  = self.token_count(PlayerColor.RED)
        blue_tok = self.token_count(PlayerColor.BLUE)
        lines.append(f"Tokens — RED: {red_tok}  BLUE: {blue_tok}")
        return "\n".join(lines)