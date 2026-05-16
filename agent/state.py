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

# Sign convention used throughout the agent: positive = Red, negative = Blue.
RED = 1
BLUE = -1

# Cardinal directions as (delta-row, delta-column) and matching Direction enums.
# Order: Up, Down, Left, Right.
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_NAMES = [Direction.Up, Direction.Down, Direction.Left, Direction.Right]
# Equivalent offsets in flat (r * 8 + c) board indexing.
DIR_OFFSETS = [dr * BOARD_N + dc for dr, dc in DIRS]

# Largest height we precompute Zobrist hash entries for. A stack can in
# principle grow by merging up to 12 friendly tokens; 24 leaves comfortable
# headroom for unusual game states.
MAX_HEIGHT_HASH = 24


# ---------------------------------------------------------------- Zobrist tables

# Random number generator seeded with a fixed constant so hashes are stable
# across runs (useful for debugging and for deterministic test replays).
_zobrist_rng = random.Random(0xC45CADE)

# Three-dimensional table: zobrist_table[colour][height][cell_index].
# colour is 0 for Red, 1 for Blue. Height starts at 1 (height 0 = empty cell
# contributes nothing, handled separately below).
_zobrist_table = [
    [[_zobrist_rng.getrandbits(64) for _ in range(NCELLS)] for _ in range(MAX_HEIGHT_HASH + 1)]
    for _ in range(2)
]

# Separate key XORed into the running hash whenever it becomes Blue's turn.
# This ensures positions with the same pieces but different sides to move
# get distinct hashes (which matters for the transposition table).
_zobrist_turn_key = _zobrist_rng.getrandbits(64)


def _zobrist_piece_key(cell_value: int, idx: int) -> int:
    """Zobrist contribution from one (signed) cell value at one cell index."""
    if cell_value == 0:
        return 0
    if cell_value > 0:
        return _zobrist_table[0][cell_value][idx]      # Red
    return _zobrist_table[1][-cell_value][idx]         # Blue


# ---------------------------------------------------------------- Undo record

@dataclass(slots=True)
class Undo:
    """
    Information needed to roll back exactly one applied action.

    Stored as a stack inside State.history -- one entry pushed per apply()
    call, one entry popped per undo() call. Search makes heavy use of this
    so we avoid the per-node cost of copying the entire board.
    """
    changes: list[tuple[int, int]]      # (cell_index, previous_value) for every mutated cell
    was_placement: bool                 # True if this action was a PLACE (increments placement_count)
    pushed_position: bool               # True if this position got added to pos_history


# ---------------------------------------------------------------- State

class State:
    """
    Mutable game state with cheap apply/undo support.

    Mirrors the referee's Board exactly. Tested for equivalence by playing
    matched random-action sequences on both and asserting per-cell equality
    after every action.
    """

    __slots__ = ('board', 'turn', 'placement_count', 'play_turn_count',
                 'hash', 'history', 'pos_history', '_pos_counts')

    def __init__(self) -> None:
        # Flat list of 64 signed ints: sign = colour, abs = height, 0 = empty.
        self.board: list[int] = [0] * NCELLS
        self.turn: int = RED                            # side to move
        self.placement_count: int = 0                   # number of PLACE actions played so far
        self.play_turn_count: int = 0                   # number of non-placement actions played
        self.hash: int = 0                              # running Zobrist hash
        self.history: list[Undo] = []                   # apply/undo stack
        # Play-phase position log for threefold-repetition detection.
        self.pos_history: list[int] = []
        # O(1) repetition lookup: hash -> count of occurrences in pos_history.
        self._pos_counts: dict[int, int] = {}

    # ----------------------------------------------------------- copying

    def clone(self) -> 'State':
        """
        Deep copy. Used rarely -- search prefers apply/undo on the original
        state. We do not copy `history` since callers of clone typically
        want a fresh undo stack.
        """
        copy = State()
        copy.board = self.board.copy()
        copy.turn = self.turn
        copy.placement_count = self.placement_count
        copy.play_turn_count = self.play_turn_count
        copy.hash = self.hash
        copy.pos_history = self.pos_history.copy()
        copy._pos_counts = self._pos_counts.copy()
        return copy

    # ----------------------------------------------------------- queries

    @property
    def is_placement_phase(self) -> bool:
        """True if we're still in the opening 8-turn placement phase."""
        return self.placement_count < PLACEMENT_TURNS

    def token_count(self, color: int) -> int:
        """Total tokens (sum of heights) belonging to the given side."""
        if color == RED:
            return sum(v for v in self.board if v > 0)
        return -sum(v for v in self.board if v < 0)

    def stack_count(self, color: int) -> int:
        """Number of (non-empty) stacks belonging to the given side."""
        if color == RED:
            return sum(1 for v in self.board if v > 0)
        return sum(1 for v in self.board if v < 0)

    # ----------------------------------------------------------- coord helpers

    @staticmethod
    def coord_to_idx(coord: Coord) -> int:
        """Flatten a (row, col) Coord into a 0-63 board index."""
        return coord.r * BOARD_N + coord.c

    @staticmethod
    def idx_to_coord(idx: int) -> Coord:
        """Inverse of coord_to_idx."""
        return Coord(idx // BOARD_N, idx % BOARD_N)

    @staticmethod
    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < BOARD_N and 0 <= c < BOARD_N

    # ----------------------------------------------------------- mutation primitive

    def _set_cell(self, idx: int, new_value: int) -> None:
        """
        Write a new value to one cell, keeping the Zobrist hash in sync
        incrementally. This is the single mutation primitive -- every
        change to self.board goes through here.
        """
        old_value = self.board[idx]
        if old_value == new_value:
            return
        # XOR out the old piece key, XOR in the new one.
        self.hash ^= _zobrist_piece_key(old_value, idx)
        self.hash ^= _zobrist_piece_key(new_value, idx)
        self.board[idx] = new_value

    # ----------------------------------------------------------- repetition tracking

    def _push_position(self) -> None:
        """Record the current position hash for threefold-repetition counting."""
        position_hash = self.hash
        self.pos_history.append(position_hash)
        self._pos_counts[position_hash] = self._pos_counts.get(position_hash, 0) + 1

    def _pop_position(self) -> None:
        """Undo the most recent _push_position."""
        position_hash = self.pos_history.pop()
        count = self._pos_counts[position_hash]
        if count == 1:
            del self._pos_counts[position_hash]
        else:
            self._pos_counts[position_hash] = count - 1

    def is_threefold(self) -> bool:
        """True if the current position has appeared >= 3 times in the play phase."""
        return self._pos_counts.get(self.hash, 0) >= 3

    # ----------------------------------------------------------- apply / undo

    def apply(self, action: Action) -> Undo:
        """
        Apply ``action`` to this state and return an Undo token.

        Caller is responsible for passing only legal actions; we do not
        validate against game rules here (the referee does that). The
        returned Undo token must eventually be reversed by undo().
        """
        changes: list[tuple[int, int]] = []
        was_placement = False

        # ---------- PLACE ----------
        if isinstance(action, PlaceAction):
            idx = self.coord_to_idx(action.coord)
            changes.append((idx, self.board[idx]))
            # Sign of turn picks the colour; magnitude is the initial height.
            self._set_cell(idx, self.turn * INITIAL_STACK_HEIGHT)
            was_placement = True
            self.placement_count += 1

        # ---------- MOVE (relocate or merge) ----------
        elif isinstance(action, MoveAction):
            src_idx = self.coord_to_idx(action.coord)
            dr, dc = _direction_to_drdc(action.direction)
            dest_idx = (action.coord.r + dr) * BOARD_N + (action.coord.c + dc)
            src_value = self.board[src_idx]
            dest_value = self.board[dest_idx]
            changes.append((src_idx, src_value))
            changes.append((dest_idx, dest_value))
            if dest_value == 0:
                # Relocate: source moves to empty dest.
                self._set_cell(dest_idx, src_value)
                self._set_cell(src_idx, 0)
            else:
                # Merge: heights sum (both have the same sign because friendly).
                self._set_cell(dest_idx, src_value + dest_value)
                self._set_cell(src_idx, 0)
            self.play_turn_count += 1

        # ---------- EAT (replace enemy, discard their tokens) ----------
        elif isinstance(action, EatAction):
            src_idx = self.coord_to_idx(action.coord)
            dr, dc = _direction_to_drdc(action.direction)
            dest_idx = (action.coord.r + dr) * BOARD_N + (action.coord.c + dc)
            src_value = self.board[src_idx]
            dest_value = self.board[dest_idx]
            changes.append((src_idx, src_value))
            changes.append((dest_idx, dest_value))
            # Attacker moves to target cell; enemy tokens are destroyed
            # (attacker's height does not absorb them).
            self._set_cell(dest_idx, src_value)
            self._set_cell(src_idx, 0)
            self.play_turn_count += 1

        # ---------- CASCADE (delegate to helper; non-trivial) ----------
        elif isinstance(action, CascadeAction):
            self._apply_cascade(action, changes)
            self.play_turn_count += 1

        else:
            raise ValueError(f'Unknown action {action!r}')

        # Side to move flips; Zobrist turn key gets XORed in.
        self.turn = -self.turn
        self.hash ^= _zobrist_turn_key

        # Threefold-repetition counts the play phase only (placement excluded).
        pushed_position = False
        if not self.is_placement_phase:
            self._push_position()
            pushed_position = True

        undo = Undo(changes=changes, was_placement=was_placement, pushed_position=pushed_position)
        self.history.append(undo)
        return undo

    def undo(self) -> None:
        """
        Reverse the most recent apply(). Must be paired 1:1 with apply.
        """
        if not self.history:
            raise IndexError('Nothing to undo')

        undo = self.history.pop()

        # Reverse the side-effects in the opposite order they were applied.
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
        Apply a CASCADE action.

        Algorithm (matches the rules document exactly):
          1. Remove the source stack of height h.
          2. For each step i in 1..h along the cascade direction:
             * If off-board: that token is discarded.
             * If empty: place a height-1 token of the cascader's colour.
             * If occupied: push the stack one cell further (recursively
               pushing whatever is behind it; stacks pushed off the edge are
               eliminated). Then place the height-1 token on the now-empty cell.

        We use a working dict to model the in-progress state so the recursive
        push logic doesn't need to commit intermediate writes to the board.
        After the cascade resolves we commit all changes at once.
        """
        source = action.coord
        source_idx = source.r * BOARD_N + source.c
        cascade_height = abs(self.board[source_idx])
        cascader_sign = self.turn        # the colour of the cascading player
        dr, dc = _direction_to_drdc(action.direction)

        # original[idx] = the pre-cascade value (for the Undo record);
        # working[idx]  = the in-progress simulated value during cascade.
        original_values: dict[int, int] = {}
        working_values: dict[int, int] = {}

        def get(idx: int) -> int:
            """Read the in-progress value, falling back to the real board."""
            if idx in working_values:
                return working_values[idx]
            return self.board[idx]

        def write(idx: int, value: int) -> None:
            """Stage a value, recording the original value on first touch."""
            if idx not in original_values:
                original_values[idx] = self.board[idx]
            working_values[idx] = value

        # Step 1: the source stack is removed.
        write(source_idx, 0)

        # Step 2: walk the cascade path.
        for step in range(1, cascade_height + 1):
            target_r = source.r + dr * step
            target_c = source.c + dc * step
            if not (0 <= target_r < BOARD_N and 0 <= target_c < BOARD_N):
                # Token falls off the board.
                continue
            target_idx = target_r * BOARD_N + target_c
            # If something's in the way, push the chain ahead one cell.
            if get(target_idx) != 0:
                self._push_chain(target_idx, dr, dc, get, write)
            # Place a height-1 token of the cascader's colour.
            write(target_idx, cascader_sign)

        # Commit working state to the real board, recording each change in the
        # Undo log via the standard _set_cell path (so the hash stays in sync).
        for idx, original_value in original_values.items():
            new_value = working_values[idx]
            if original_value != new_value:
                changes.append((idx, original_value))
                self._set_cell(idx, new_value)

    @staticmethod
    def _push_chain(idx: int, dr: int, dc: int, get, write) -> None:
        """
        Push the stack at ``idx`` one cell in direction (dr, dc), recursively
        pushing anything in the way. Stacks shoved off the board are eliminated.

        Operates on the cascade-in-progress working state via the get/write
        callbacks rather than directly on self.board.
        """
        cell_value = get(idx)
        if cell_value == 0:
            return
        r = idx // BOARD_N
        c = idx % BOARD_N
        new_r = r + dr
        new_c = c + dc
        if not (0 <= new_r < BOARD_N and 0 <= new_c < BOARD_N):
            # Pushed off the edge: this stack is eliminated.
            write(idx, 0)
            return
        dest_idx = new_r * BOARD_N + new_c
        # Recursively push whatever is at the destination before we land.
        if get(dest_idx) != 0:
            State._push_chain(dest_idx, dr, dc, get, write)
        # Now move the stack from idx to dest_idx.
        write(dest_idx, cell_value)
        write(idx, 0)

    # ----------------------------------------------------------- game-end checks

    def game_over(self) -> bool:
        """
        True if the current position is terminal under any of the four
        end conditions, in spec-mandated precedence order:
          1. Elimination
          2. Turn-limit reached
          3. Threefold repetition (draw)
          4. Stalemate -- side to move has no legal action (draw)

        Returns False during the placement phase, which can't be terminal.
        """
        if self.is_placement_phase:
            return False
        if self.token_count(RED) == 0:
            return True
        if self.token_count(BLUE) == 0:
            return True
        if self.play_turn_count >= MAX_PLAY_TURNS:
            return True
        if self.is_threefold():
            return True
        if not self._has_any_legal_action():
            return True
        return False

    def winner(self) -> int | None:
        """
        Outcome of a terminal position.

        Returns +1 if Red wins, -1 if Blue wins, 0 for a draw, or None if
        the game isn't over yet. Precedence matches game_over() so this
        is the canonical interpreter for terminal states.
        """
        if self.is_placement_phase:
            return None
        red_tokens = self.token_count(RED)
        blue_tokens = self.token_count(BLUE)

        # Elimination (handle the degenerate "both at zero" case as a draw).
        if red_tokens == 0 and blue_tokens == 0:
            return 0
        if red_tokens == 0:
            return BLUE
        if blue_tokens == 0:
            return RED

        # Draws by repetition or stalemate.
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
        """Imported lazily to avoid a circular import with the moves module."""
        from .moves import has_any_legal_action
        return has_any_legal_action(self)


# ---------------------------------------------------------------- helpers

# Lookup table converting a Direction enum to its (dr, dc) tuple.
# Constructed once so the hot path doesn't allocate.
_DIRECTION_TO_DRDC = {
    Direction.Up: (-1, 0),
    Direction.Down: (1, 0),
    Direction.Left: (0, -1),
    Direction.Right: (0, 1),
}


def _direction_to_drdc(direction: Direction) -> tuple[int, int]:
    """Decode a cardinal Direction into a (delta-row, delta-column) pair."""
    return _DIRECTION_TO_DRDC[direction]