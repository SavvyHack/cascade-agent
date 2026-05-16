from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Iterable
from referee.game import Action, PlaceAction, MoveAction, EatAction, CascadeAction, Coord, Direction, PlayerColor

BOARD_N = 8
NCELLS = BOARD_N * BOARD_N
PLACEMENT_TURNS = 8
MAX_PLAY_TURNS = 300
INITIAL_STACK_HEIGHT = 3
RED = 1
BLUE = -1
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_NAMES = [Direction.Up, Direction.Down, Direction.Left, Direction.Right]
DIR_OFFSETS = [dr * BOARD_N + dc for dr, dc in DIRS]
MAX_HEIGHT_HASH = 24

_rng = random.Random(0xC45CADE)
_ZOBRIST = [
    [[_rng.getrandbits(64) for _ in range(NCELLS)] for _ in range(MAX_HEIGHT_HASH + 1)]
    for _ in range(2)
]
_ZOBRIST_TURN = _rng.getrandbits(64)


def _piece_key(value: int, idx: int) -> int:
    if value == 0:
        return 0
    if value > 0:
        return _ZOBRIST[0][value][idx]
    return _ZOBRIST[1][-value][idx]


@dataclass(slots=True)
class Undo:
    changes: list[tuple[int, int]]
    was_placement: bool
    pushed_position: bool


class State:
    __slots__ = ('board', 'turn', 'placement_count', 'play_turn_count',
                 'hash', 'history', 'pos_history', '_pos_counts')

    def __init__(self) -> None:
        self.board: list[int] = [0] * NCELLS
        self.turn: int = RED
        self.placement_count: int = 0
        self.play_turn_count: int = 0
        self.hash: int = 0
        self.history: list[Undo] = []
        self.pos_history: list[int] = []
        self._pos_counts: dict[int, int] = {}

    def clone(self) -> 'State':
        s = State()
        s.board = self.board.copy()
        s.turn = self.turn
        s.placement_count = self.placement_count
        s.play_turn_count = self.play_turn_count
        s.hash = self.hash
        s.pos_history = self.pos_history.copy()
        s._pos_counts = self._pos_counts.copy()
        return s

    @property
    def is_placement_phase(self) -> bool:
        return self.placement_count < PLACEMENT_TURNS

    def token_count(self, color: int) -> int:
        if color == RED:
            return sum(v for v in self.board if v > 0)
        return -sum(v for v in self.board if v < 0)

    def stack_count(self, color: int) -> int:
        if color == RED:
            return sum(1 for v in self.board if v > 0)
        return sum(1 for v in self.board if v < 0)

    @staticmethod
    def coord_to_idx(coord: Coord) -> int:
        return coord.r * BOARD_N + coord.c

    @staticmethod
    def idx_to_coord(idx: int) -> Coord:
        return Coord(idx // BOARD_N, idx % BOARD_N)

    @staticmethod
    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < BOARD_N and 0 <= c < BOARD_N

    def _set_cell(self, idx: int, new_value: int) -> None:
        old_value = self.board[idx]
        if old_value == new_value:
            return
        self.hash ^= _piece_key(old_value, idx)
        self.hash ^= _piece_key(new_value, idx)
        self.board[idx] = new_value

    def _push_position(self) -> None:
        h = self.hash
        self.pos_history.append(h)
        self._pos_counts[h] = self._pos_counts.get(h, 0) + 1

    def _pop_position(self) -> None:
        h = self.pos_history.pop()
        c = self._pos_counts[h]
        if c == 1:
            del self._pos_counts[h]
        else:
            self._pos_counts[h] = c - 1

    def is_threefold(self) -> bool:
        return self._pos_counts.get(self.hash, 0) >= 3

    def apply(self, action: Action) -> Undo:
        changes: list[tuple[int, int]] = []
        was_placement = False

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
            src_val = self.board[src_idx]
            dest_val = self.board[dest_idx]
            changes.append((src_idx, src_val))
            changes.append((dest_idx, dest_val))
            if dest_val == 0:
                self._set_cell(dest_idx, src_val)
                self._set_cell(src_idx, 0)
            else:
                self._set_cell(dest_idx, src_val + dest_val)
                self._set_cell(src_idx, 0)
            self.play_turn_count += 1

        elif isinstance(action, EatAction):
            src_idx = self.coord_to_idx(action.coord)
            dr, dc = _direction_to_drdc(action.direction)
            dest_idx = (action.coord.r + dr) * BOARD_N + (action.coord.c + dc)
            src_val = self.board[src_idx]
            dest_val = self.board[dest_idx]
            changes.append((src_idx, src_val))
            changes.append((dest_idx, dest_val))
            self._set_cell(dest_idx, src_val)
            self._set_cell(src_idx, 0)
            self.play_turn_count += 1

        elif isinstance(action, CascadeAction):
            self._apply_cascade(action, changes)
            self.play_turn_count += 1

        else:
            raise ValueError(f'Unknown action {action!r}')

        self.turn = -self.turn
        self.hash ^= _ZOBRIST_TURN

        pushed = False
        if not self.is_placement_phase:
            self._push_position()
            pushed = True

        undo = Undo(changes=changes, was_placement=was_placement, pushed_position=pushed)
        self.history.append(undo)
        return undo

    def undo(self) -> None:
        if not self.history:
            raise IndexError('Nothing to undo')
        undo = self.history.pop()
        if undo.pushed_position:
            self._pop_position()
        self.turn = -self.turn
        self.hash ^= _ZOBRIST_TURN
        for idx, prev in reversed(undo.changes):
            self._set_cell(idx, prev)
        if undo.was_placement:
            self.placement_count -= 1
        else:
            self.play_turn_count -= 1

    def _apply_cascade(self, action: CascadeAction, changes: list[tuple[int, int]]) -> None:
        src = action.coord
        src_idx = src.r * BOARD_N + src.c
        h = abs(self.board[src_idx])
        cascader = self.turn
        dr, dc = _direction_to_drdc(action.direction)

        original: dict[int, int] = {}
        working: dict[int, int] = {}

        def get(idx: int) -> int:
            if idx in working:
                return working[idx]
            return self.board[idx]

        def write(idx: int, val: int) -> None:
            if idx not in original:
                original[idx] = self.board[idx]
            working[idx] = val

        write(src_idx, 0)

        for i in range(1, h + 1):
            tr = src.r + dr * i
            tc = src.c + dc * i
            if not (0 <= tr < BOARD_N and 0 <= tc < BOARD_N):
                continue
            t_idx = tr * BOARD_N + tc
            if get(t_idx) != 0:
                self._push_chain(t_idx, dr, dc, get, write)
            write(t_idx, cascader)

        for idx, old_val in original.items():
            new_val = working[idx]
            if old_val != new_val:
                changes.append((idx, old_val))
                self._set_cell(idx, new_val)

    @staticmethod
    def _push_chain(idx: int, dr: int, dc: int, get, write) -> None:
        cell = get(idx)
        if cell == 0:
            return
        r = idx // BOARD_N
        c = idx % BOARD_N
        nr = r + dr
        nc = c + dc
        if not (0 <= nr < BOARD_N and 0 <= nc < BOARD_N):
            write(idx, 0)
            return
        dest_idx = nr * BOARD_N + nc
        if get(dest_idx) != 0:
            State._push_chain(dest_idx, dr, dc, get, write)
        write(dest_idx, cell)
        write(idx, 0)

    def game_over(self) -> bool:
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
        if self.is_placement_phase:
            return None
        red = self.token_count(RED)
        blue = self.token_count(BLUE)
        if red == 0 and blue == 0:
            return 0
        if red == 0:
            return BLUE
        if blue == 0:
            return RED
        if self.is_threefold():
            return 0
        if not self._has_any_legal_action():
            return 0
        if self.play_turn_count >= MAX_PLAY_TURNS:
            if red > blue:
                return RED
            if blue > red:
                return BLUE
            return 0
        return None

    def _has_any_legal_action(self) -> bool:
        from .moves import has_any_legal_action
        return has_any_legal_action(self)


_DIRECTION_TO_DRDC = {
    Direction.Up: (-1, 0),
    Direction.Down: (1, 0),
    Direction.Left: (0, -1),
    Direction.Right: (0, 1),
}


def _direction_to_drdc(direction: Direction) -> tuple[int, int]:
    return _DIRECTION_TO_DRDC[direction]