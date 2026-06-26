from __future__ import annotations
from referee.game import Action, PlaceAction, MoveAction, EatAction, CascadeAction, Coord
from .state import State, BOARD_N, NCELLS, DIRS, DIR_NAMES, RED, BLUE


# Precomputed (neighbour_idx, direction_index) per cell.
_adjacent_cells: list[list[tuple[int, int]]] = []
for r in range(BOARD_N):
    for c in range(BOARD_N):
        neighbours = []
        for dir_index, (dr, dc) in enumerate(DIRS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_N and 0 <= nc < BOARD_N:
                neighbours.append((nr * BOARD_N + nc, dir_index))
        _adjacent_cells.append(neighbours)


def idx_to_coord(idx: int) -> Coord:
    return Coord(idx // BOARD_N, idx % BOARD_N)


def gen_placement_moves(state: State) -> list[PlaceAction]:
    """Every legal PLACE action for the side to move during the placement phase."""
    board = state.board
    opponent_sign = -state.turn
    is_first_placement = state.placement_count == 0

    placements: list[PlaceAction] = []
    for idx in range(NCELLS):
        if board[idx] != 0:
            continue
        if not is_first_placement:
            adjacent_to_opponent = False
            for neighbour_idx, _ in _adjacent_cells[idx]:
                neighbour = board[neighbour_idx]
                if (neighbour > 0 and opponent_sign == RED) or \
                   (neighbour < 0 and opponent_sign == BLUE):
                    adjacent_to_opponent = True
                    break
            if adjacent_to_opponent:
                continue
        placements.append(PlaceAction(idx_to_coord(idx)))
    return placements


def _cascade_has_effect(board: list[int], src_idx: int,
                        dr: int, dc: int, height: int) -> bool:
    """
    True if a cascade from src_idx in (dr,dc) by `height` tokens has any
    meaningful effect: hits a stack along the way, or lands entirely on the
    board (splitting our stack into smaller pieces is a real action).

    Excludes only cascades that walk straight off the board without hitting
    anything -- pure self-elimination with no opponent contact.
    """
    sr = src_idx // BOARD_N
    sc = src_idx % BOARD_N
    walked_on_board = 0
    for step in range(1, height + 1):
        nr = sr + dr * step
        nc = sc + dc * step
        if not (0 <= nr < BOARD_N and 0 <= nc < BOARD_N):
            break
        walked_on_board += 1
        if board[nr * BOARD_N + nc] != 0:
            return True
    # Whole cascade stayed on the board: every token lands -> real effect.
    return walked_on_board == height


def gen_play_moves(state: State) -> list[Action]:
    """Every legal play-phase action for the side to move."""
    board = state.board
    me = state.turn
    actions: list[Action] = []

    for idx in range(NCELLS):
        cell = board[idx]
        if me == RED:
            if cell <= 0:
                continue
            our_height = cell
        else:
            if cell >= 0:
                continue
            our_height = -cell

        coord = idx_to_coord(idx)

        # MOVE + EAT per cardinal neighbour.
        for neighbour_idx, dir_index in _adjacent_cells[idx]:
            neighbour = board[neighbour_idx]
            direction = DIR_NAMES[dir_index]
            if neighbour == 0:
                actions.append(MoveAction(coord, direction))
            elif (neighbour > 0) == (me == RED):
                actions.append(MoveAction(coord, direction))
            else:
                target_height = abs(neighbour)
                if our_height >= target_height:
                    actions.append(EatAction(coord, direction))

        # CASCADE per cardinal direction (height >= 2), filtered by effect.
        if our_height >= 2:
            for dir_index, direction in enumerate(DIR_NAMES):
                dr, dc = DIRS[dir_index]
                if _cascade_has_effect(board, idx, dr, dc, our_height):
                    actions.append(CascadeAction(coord, direction))

    return actions


def gen_all_moves(state: State) -> list[Action]:
    if state.is_placement_phase:
        return gen_placement_moves(state)
    return gen_play_moves(state)


def has_any_legal_action(state: State) -> bool:
    """Fast existence check (returns at the first legal action found)."""
    board = state.board

    if state.is_placement_phase:
        opponent_sign = -state.turn
        is_first_placement = state.placement_count == 0
        for idx in range(NCELLS):
            if board[idx] != 0:
                continue
            if is_first_placement:
                return True
            adjacent_to_opponent = False
            for neighbour_idx, _ in _adjacent_cells[idx]:
                neighbour = board[neighbour_idx]
                if (neighbour > 0 and opponent_sign == RED) or \
                   (neighbour < 0 and opponent_sign == BLUE):
                    adjacent_to_opponent = True
                    break
            if not adjacent_to_opponent:
                return True
        return False

    me = state.turn
    for idx in range(NCELLS):
        cell = board[idx]
        if me == RED:
            if cell <= 0:
                continue
            our_height = cell
        else:
            if cell >= 0:
                continue
            our_height = -cell

        for neighbour_idx, _ in _adjacent_cells[idx]:
            neighbour = board[neighbour_idx]
            if neighbour == 0:
                return True
            if (neighbour > 0) == (me == RED):
                return True
            if our_height >= abs(neighbour):
                return True

        # A height-2+ stack always has at least one effect-bearing cascade if
        # there's anything on the board besides it, but rather than try to
        # reason about that here we just say: cascades exist for height >= 2.
        if our_height >= 2:
            return True

    return False
