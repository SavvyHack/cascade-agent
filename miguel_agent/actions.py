from referee.game import PlayerColor, Board, GamePhase, Coord, PlaceAction, \
    MoveAction, EatAction, CascadeAction, CARDINAL_DIRECTIONS
from .game_state import COORD_TABLE, DIRS, BOARD_N

# Pre-built coord cache. Avoids repeated Coord(r,c) construction in hot loops
_COORD_CACHE: dict[tuple[int, int], Coord] = {
    (r, c): Coord(r, c) for r in range(BOARD_N) for c in range(BOARD_N)
}


def _in_bounds(r: int, c: int) -> bool:
    """True if (r, c) is on the board."""
    return 0 <= r < BOARD_N and 0 <= c < BOARD_N


def legal_place_actions(board: Board, color: PlayerColor) -> list[PlaceAction]:
    """Return all legal PLACE actions for color."""
    opponent = color.opponent
    first_placement = (board._placement_count == 0)
    actions = []
    cc = _COORD_CACHE # local alias for speed

    for coord in COORD_TABLE:
        if not board[coord].is_empty:
            continue
        if first_placement:
            actions.append(PlaceAction(coord))
            continue

        # Must not be adjacent to an opponent token
        adj = False
        for _, dr, dc in DIRS:
            nr, nc = coord.r + dr, coord.c + dc
            if _in_bounds(nr, nc) and board[cc[(nr, nc)]].color == opponent:
                adj = True
                break

        if not adj:
            actions.append(PlaceAction(coord))

    return actions


def legal_play_actions(board: Board, color: PlayerColor) -> list:
    """Return all legal PLAY actions (Move / Eat / Cascade) for color."""
    opponent = color.opponent
    actions = []
    cc = _COORD_CACHE # local alias for speed

    for coord in COORD_TABLE:
        cell = board[coord]
        if cell.color != color:
            continue
        h = cell.height

        for direction in CARDINAL_DIRECTIONS:
            dr, dc = direction.r, direction.c
            nr, nc = coord.r + dr, coord.c + dc
            if not _in_bounds(nr, nc):
                continue
            dest = board[cc[(nr, nc)]]

            if dest.is_empty or dest.color == color:
                actions.append(MoveAction(coord, direction))
            elif dest.color == opponent and h >= dest.height:
                actions.append(EatAction(coord, direction))

        if h >= 2:
            for direction in CARDINAL_DIRECTIONS:
                dr, dc = direction.r, direction.c
                has_effect = False
                for step in range(1, h + 1):
                    nr, nc = coord.r + dr * step, coord.c + dc * step
                    if not _in_bounds(nr, nc):
                        break
                    if not board[cc[(nr, nc)]].is_empty:
                        has_effect = True
                        break
                else:
                    has_effect = True # cascade reaches empty space beyond board edge
                if has_effect:
                    actions.append(CascadeAction(coord, direction))

    return actions


def legal_actions(board: Board, color: PlayerColor) -> list:
    """Dispatch to placement or play action generator based on game phase."""
    if board.phase == GamePhase.PLACEMENT:
        return legal_place_actions(board, color)
    return legal_play_actions(board, color)