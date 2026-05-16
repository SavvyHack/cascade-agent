from __future__ import annotations
from referee.game import Action, PlaceAction, MoveAction, EatAction, CascadeAction, Coord
from .state import State, BOARD_N, NCELLS, DIRS, DIR_NAMES, RED, BLUE


# Precomputed adjacency table: for each cell index, a list of
# (neighbour_index, direction_index) tuples covering the up-to-4 cardinal
# neighbours. Built once at module load so move generation never recomputes
# bounds checks or coordinate arithmetic.
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
    """Convert a flat board index (0-63) back to a referee Coord(r, c)."""
    return Coord(idx // BOARD_N, idx % BOARD_N)


def gen_placement_moves(state: State) -> list[PlaceAction]:
    """
    Generate every legal PLACE action for the side to move during the
    placement phase.

    The only rule beyond "must be empty": after the very first placement of
    the game, you may not place adjacent to any opponent stack.
    """
    board = state.board
    opponent_sign = -state.turn
    is_first_placement = state.placement_count == 0

    placements: list[PlaceAction] = []
    for idx in range(NCELLS):
        # Must be empty.
        if board[idx] != 0:
            continue

        # Adjacency-to-opponent rule applies after the first placement.
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


def gen_play_moves(state: State) -> list[Action]:
    """
    Generate every legal play-phase action for the side to move.

    Per friendly stack we consider:
      * MOVE to an empty neighbour (relocate)
      * MOVE onto a friendly neighbour (merge)
      * EAT an adjacent enemy of equal-or-smaller height
      * CASCADE in any of the four cardinal directions (if height >= 2)
    """
    board = state.board
    me = state.turn
    actions: list[Action] = []

    for idx in range(NCELLS):
        cell = board[idx]

        # Skip empty cells and enemy stacks; we only generate from our own.
        if me == RED:
            if cell <= 0:
                continue
            our_height = cell
        else:
            if cell >= 0:
                continue
            our_height = -cell

        coord = idx_to_coord(idx)

        # MOVE and EAT actions from each cardinal neighbour.
        for neighbour_idx, dir_index in _adjacent_cells[idx]:
            neighbour = board[neighbour_idx]
            direction = DIR_NAMES[dir_index]

            if neighbour == 0:
                # Relocate into an empty cell.
                actions.append(MoveAction(coord, direction))
            elif (neighbour > 0) == (me == RED):
                # Merge with a friendly stack.
                actions.append(MoveAction(coord, direction))
            else:
                # Enemy stack: EAT is legal only if our height >= theirs.
                target_height = abs(neighbour)
                if our_height >= target_height:
                    actions.append(EatAction(coord, direction))

        # CASCADE actions: any cardinal direction, available if height >= 2.
        # We don't filter by whether the cascade is "good" -- the search
        # handles that. We just generate all four.
        if our_height >= 2:
            for dir_index, direction in enumerate(DIR_NAMES):
                actions.append(CascadeAction(coord, direction))

    return actions


def gen_all_moves(state: State) -> list[Action]:
    """
    Dispatch to the appropriate generator based on game phase.

    Used everywhere except the inner fast-path in has_any_legal_action.
    """
    if state.is_placement_phase:
        return gen_placement_moves(state)
    return gen_play_moves(state)


def has_any_legal_action(state: State) -> bool:
    """
    Fast existence check: does the side to move have *any* legal action?

    Used for stalemate detection at terminal nodes. Returns as soon as the
    first legal action is found, without building the full move list -- this
    matters at depth-limited search nodes where we just need a yes/no.
    """
    board = state.board

    # ---------- placement phase ----------
    if state.is_placement_phase:
        opponent_sign = -state.turn
        is_first_placement = state.placement_count == 0
        for idx in range(NCELLS):
            if board[idx] != 0:
                continue
            # First placement: any empty cell works.
            if is_first_placement:
                return True
            # Otherwise check the adjacency rule.
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

    # ---------- play phase ----------
    me = state.turn
    for idx in range(NCELLS):
        cell = board[idx]

        # Only our own stacks can act.
        if me == RED:
            if cell <= 0:
                continue
            our_height = cell
        else:
            if cell >= 0:
                continue
            our_height = -cell

        # If any neighbour is empty (MOVE), friendly (MERGE), or eatable (EAT),
        # we have a legal action -- return immediately.
        for neighbour_idx, _ in _adjacent_cells[idx]:
            neighbour = board[neighbour_idx]
            if neighbour == 0:
                return True
            if (neighbour > 0) == (me == RED):
                return True
            if our_height >= abs(neighbour):
                return True

        # CASCADE is always legal for height >= 2 (it can fall off the board
        # but that's still a legal action).
        if our_height >= 2:
            return True

    return False