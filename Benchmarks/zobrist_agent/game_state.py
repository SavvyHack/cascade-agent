from referee.game import PlayerColor, Board, Coord, Direction

BOARD_N = 8
EMPTY = 0
RED_C = 1
BLUE_C = 2

COORD_TABLE = [Coord(r, c) for r, c in [divmod(i, BOARD_N) for i in range(BOARD_N * BOARD_N)]]
DIRS = [(d, d.r, d.c) for d in Direction] # (Direction, dr, dc) triples


def board_to_tuple(board: Board) -> tuple:
    """Convert Board to a flat tuple suitable for hashing."""
    state = [EMPTY] * (BOARD_N * BOARD_N)
    for i, coord in enumerate(COORD_TABLE):
        cell = board[coord]
        if cell.is_stack:
            colour, _ = cell
            c = RED_C if colour == PlayerColor.RED else BLUE_C
            state[i] = (c, cell.height)
    return tuple(state)