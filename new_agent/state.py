from __future__ import annotations
import random
from dataclasses import dataclass
from referee.game import Action, PlaceAction, MoveAction, EatAction, CascadeAction, Coord, Direction


# ---------------------------------------------------------------- Game constants

BOARD_N = 8
NCELLS = BOARD_N * BOARD_N
PLACEMENT_TURNS = 8
MAX_PLAY_TURNS = 300
INITIAL_STACK_HEIGHT = 3

# Sign convention: positive = Red, negative = Blue.
RED = 1
BLUE = -1

# Cardinal directions (Up, Down, Left, Right) as deltas and Direction enums.
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_NAMES = [Direction.Up, Direction.Down, Direction.Left, Direction.Right]
DIR_OFFSETS = [dr * BOARD_N + dc for dr, dc in DIRS]

# Cap on the height for which we precompute Zobrist entries; merging twelve
# friendly tokens is the natural max, so 24 leaves headroom for unusual play.
MAX_HEIGHT_HASH = 24


# ---------------------------------------------------------------- Zobrist tables

# Seeded RNG so hashes stay stable across runs.
_zobrist_rng = random.Random(0xC45CADE)

# zobrist_table[colour][height][cell_index]; colour 0 = Red, 1 = Blue.
_zobrist_table = [
    [[_zobrist_rng.getrandbits(64) for _ in range(NCELLS)] for _ in range(MAX_HEIGHT_HASH + 1)]
    for _ in range(2)
]

# XOR'd into the running hash on every side-to-move flip.
_zobrist_turn_key = _zobrist_rng.getrandbits(64)


def _zobrist_piece_key(cell_value: int, idx: int) -> int:
    """Zobrist contribution from one signed cell value at one cell index."""
    if cell_value == 0:
        return 0
    if cell_value > 0:
        return _zobrist_table[0][cell_value][idx]
    return _zobrist_table[1][-cell_value][idx]


# ---------------------------------------------------------------- Undo record

@dataclass(slots=True)
class Undo:
    """One reversible action's worth of state changes."""
    changes: list[tuple[int, int]]
    was_placement: bool
    pushed_position: bool
    red_delta: int             # change to _red_tokens this action induced
    blue_delta: int            # change to _blue_tokens this action induced


# ---------------------------------------------------------------- State

class State:
    """
    Mutable game state with cheap apply/undo, incremental Zobrist hashing,
    and O(1) per-side token counts. Mirrors the referee's Board exactly.
    """

    __slots__ = ('board', 'turn', 'placement_count', 'play_turn_count',
                 'hash', 'history', 'pos_history', '_pos_counts',
                 '_red_tokens', '_blue_tokens')

    def __init__(self) -> None:
        self.board: list[int] = [0] * NCELLS
        self.turn: int = RED
        self.placement_count: int = 0
        self.play_turn_count: int = 0
        self.hash: int = 0
        self.history: list[Undo] = []
        # Play-phase position log for threefold-repetition detection.
        self.pos_history: list[int] = []
        self._pos_counts: dict[int, int] = {}
        # Running per-side totals so token_count is O(1).
        self._red_tokens: int = 0
        self._blue_tokens: int = 0

    # ----------------------------------------------------------- copying

    def clone(self) -> 'State':
        copy = State()
        copy.board = self.board.copy()
        copy.turn = self.turn
        copy.placement_count = self.placement_count
        copy.play_turn_count = self.play_turn_count
        copy.hash = self.hash
        copy.pos_history = self.pos_history.copy()
        copy._pos_counts = self._pos_counts.copy()
        copy._red_tokens = self._red_tokens
        copy._blue_tokens = self._blue_tokens
        return copy

    # ----------------------------------------------------------- queries

    @property
    def is_placement_phase(self) -> bool:
        return self.placement_count < PLACEMENT_TURNS

    def token_count(self, color: int) -> int:
        """Total tokens for the given side -- O(1)."""
        return self._red_tokens if color == RED else self._blue_tokens

    def stack_count(self, color: int) -> int:
        """Number of (non-empty) stacks belonging to the given side."""
        if color == RED:
            return sum(1 for v in self.board if v > 0)
        return sum(1 for v in self.board if v < 0)

    # ----------------------------------------------------------- coord helpers

    @staticmethod
    def coord_to_idx(coord: Coord) -> int:
        return coord.r * BOARD_N + coord.c

    @staticmethod
    def idx_to_coord(idx: int) -> Coord:
        return Coord(idx // BOARD_N, idx % BOARD_N)

    @staticmethod
    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < BOARD_N and 0 <= c < BOARD_N

    # ----------------------------------------------------------- mutation primitive

    def _set_cell(self, idx: int, new_value: int) -> None:
        """
        Single mutation primitive: writes one cell, keeps the Zobrist hash
        incrementally in sync, and maintains the per-side token totals.
        """
        old_value = self.board[idx]
        if old_value == new_value:
            return
        # Hash: XOR out old, XOR in new.
        self.hash ^= _zobrist_piece_key(old_value, idx)
        self.hash ^= _zobrist_piece_key(new_value, idx)
        # Token totals: account for the height delta on each side.
        if old_value > 0:
            self._red_tokens -= old_value
        elif old_value < 0:
            self._blue_tokens -= -old_value
        if new_value > 0:
            self._red_tokens += new_value
        elif new_value < 0:
            self._blue_tokens += -new_value
        self.board[idx] = new_value

    # ----------------------------------------------------------- repetition tracking

    def _push_position(self) -> None:
        position_hash = self.hash
        self.pos_history.append(position_hash)
        self._pos_counts[position_hash] = self._pos_counts.get(position_hash, 0) + 1

    def _pop_position(self) -> None:
        position_hash = self.pos_history.pop()
        count = self._pos_counts[position_hash]
        if count == 1:
            del self._pos_counts[position_hash]
        else:
            self._pos_counts[position_hash] = count - 1

    def is_threefold(self) -> bool:
        return self._pos_counts.get(self.hash, 0) >= 3

    # ----------------------------------------------------------- apply / undo

    def apply(self, action: Action) -> Undo:
        """Apply ``action`` and return an Undo token. Caller passes legal actions only."""
        changes: list[tuple[int, int]] = []
        was_placement = False
        red_before = self._red_tokens
        blue_before = self._blue_tokens

        if isinstance(action, PlaceAction):
            idx = self.coord_to_idx(action.coord)
            changes.append((idx, self.board[idx]))
            self._set_cell(idx, self.turn * INITIAL_STACK_HEIGHT)
            was_placement = True
            self.placement_count += 1

        elif isinstance(action, MoveAction):
            src_idx = self.coord_to_idx(action.coord)
            dr, dc = _direction_to_drdc(action.direction)
            dest_idx = (action.coord.r + dr) * BOARD_N + (action.coord.c + dc)
            src_value = self.board[src_idx]
            dest_value = self.board[dest_idx]
            changes.append((src_idx, src_value))
            changes.append((dest_idx, dest_value))
            if dest_value == 0:
                self._set_cell(dest_idx, src_value)
                self._set_cell(src_idx, 0)
            else:
                # Merge: heights sum (same sign because friendly).
                self._set_cell(dest_idx, src_value + dest_value)
                self._set_cell(src_idx, 0)
            self.play_turn_count += 1

        elif isinstance(action, EatAction):
            src_idx = self.coord_to_idx(action.coord)
            dr, dc = _direction_to_drdc(action.direction)
            dest_idx = (action.coord.r + dr) * BOARD_N + (action.coord.c + dc)
            src_value = self.board[src_idx]
            dest_value = self.board[dest_idx]
            changes.append((src_idx, src_value))
            changes.append((dest_idx, dest_value))
            # Attacker moves to target cell; enemy tokens are discarded
            # (attacker height does NOT absorb them).
            self._set_cell(dest_idx, src_value)
            self._set_cell(src_idx, 0)
            self.play_turn_count += 1

        elif isinstance(action, CascadeAction):
            self._apply_cascade(action, changes)
            self.play_turn_count += 1

        else:
            raise ValueError(f'Unknown action {action!r}')

        self.turn = -self.turn
        self.hash ^= _zobrist_turn_key

        pushed_position = False
        if not self.is_placement_phase:
            self._push_position()
            pushed_position = True

        undo = Undo(
            changes=changes,
            was_placement=was_placement,
            pushed_position=pushed_position,
            red_delta=self._red_tokens - red_before,
            blue_delta=self._blue_tokens - blue_before,
        )
        self.history.append(undo)
        return undo

    def undo(self) -> None:
        if not self.history:
            raise IndexError('Nothing to undo')

        undo = self.history.pop()

        if undo.pushed_position:
            self._pop_position()
        self.turn = -self.turn
        self.hash ^= _zobrist_turn_key
        for idx, previous_value in reversed(undo.changes):
            self._set_cell(idx, previous_value)
        if undo.was_placement:
            self.placement_count -= 1
        else:
            self.play_turn_count -= 1

    # ----------------------------------------------------------- CASCADE helper

    def _apply_cascade(self, action: CascadeAction, changes: list[tuple[int, int]]) -> None:
        """
        Apply a CASCADE per the rules:
          1. Source stack of height h is removed.
          2. For each step i in 1..h along the direction:
             - off-board: token discarded.
             - empty: place a height-1 token of the cascader's colour.
             - occupied: push the stack one cell further (recursively); stacks
               pushed off the board are eliminated. Then place the height-1 token.
        We commit via a working dict so partial state never leaks to the board.
        """
        source = action.coord
        source_idx = source.r * BOARD_N + source.c
        cascade_height = abs(self.board[source_idx])
        cascader_sign = self.turn
        dr, dc = _direction_to_drdc(action.direction)

        original_values: dict[int, int] = {}
        working_values: dict[int, int] = {}

        def get(idx: int) -> int:
            if idx in working_values:
                return working_values[idx]
            return self.board[idx]

        def write(idx: int, value: int) -> None:
            if idx not in original_values:
                original_values[idx] = self.board[idx]
            working_values[idx] = value

        write(source_idx, 0)

        for step in range(1, cascade_height + 1):
            target_r = source.r + dr * step
            target_c = source.c + dc * step
            if not (0 <= target_r < BOARD_N and 0 <= target_c < BOARD_N):
                continue   # cascade token falls off
            target_idx = target_r * BOARD_N + target_c
            if get(target_idx) != 0:
                self._push_chain(target_idx, dr, dc, get, write)
            write(target_idx, cascader_sign)

        for idx, original_value in original_values.items():
            new_value = working_values[idx]
            if original_value != new_value:
                changes.append((idx, original_value))
                self._set_cell(idx, new_value)

    @staticmethod
    def _push_chain(idx: int, dr: int, dc: int, get, write) -> None:
        """Recursively push the stack at idx one cell in (dr,dc); off-edge eliminates."""
        cell_value = get(idx)
        if cell_value == 0:
            return
        r = idx // BOARD_N
        c = idx % BOARD_N
        new_r = r + dr
        new_c = c + dc
        if not (0 <= new_r < BOARD_N and 0 <= new_c < BOARD_N):
            write(idx, 0)
            return
        dest_idx = new_r * BOARD_N + new_c
        if get(dest_idx) != 0:
            State._push_chain(dest_idx, dr, dc, get, write)
        write(dest_idx, cell_value)
        write(idx, 0)

    # ----------------------------------------------------------- game-end checks

    def game_over(self) -> bool:
        """
        Spec precedence: Elimination > Threefold > Stalemate > Turn limit.
        Always False during the placement phase.
        """
        if self.is_placement_phase:
            return False
        if self._red_tokens == 0 or self._blue_tokens == 0:
            return True
        if self.play_turn_count >= MAX_PLAY_TURNS:
            return True
        if self.is_threefold():
            return True
        if not self._has_any_legal_action():
            return True
        return False

    def winner(self) -> int | None:
        """Returns +1/-1/0 for terminal positions, or None if game ongoing."""
        if self.is_placement_phase:
            return None
        red_tokens = self._red_tokens
        blue_tokens = self._blue_tokens

        # Elimination first.
        if red_tokens == 0 and blue_tokens == 0:
            return 0
        if red_tokens == 0:
            return BLUE
        if blue_tokens == 0:
            return RED

        # Threefold and stalemate are draws.
        if self.is_threefold():
            return 0
        if not self._has_any_legal_action():
            return 0

        # Turn limit: more tokens wins, else draw.
        if self.play_turn_count >= MAX_PLAY_TURNS:
            if red_tokens > blue_tokens:
                return RED
            if blue_tokens > red_tokens:
                return BLUE
            return 0

        return None

    def _has_any_legal_action(self) -> bool:
        from .moves import has_any_legal_action
        return has_any_legal_action(self)


# ---------------------------------------------------------------- helpers

_DIRECTION_TO_DRDC = {
    Direction.Up: (-1, 0),
    Direction.Down: (1, 0),
    Direction.Left: (0, -1),
    Direction.Right: (0, 1),
}


def _direction_to_drdc(direction: Direction) -> tuple[int, int]:
    return _DIRECTION_TO_DRDC[direction]
