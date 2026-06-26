from referee.game import PlayerColor, Board, Coord
from .game_state import COORD_TABLE, DIRS
from .actions import _in_bounds

INF = float('inf')

# Heuristic weights
W_MATERIAL = 100 # token-count delta — dominates; win condition proxy
W_STACK = 3 # stack-count delta; small mobility proxy
W_CAPTURE_THREAT = 25 # enemy tokens reachable by our eat moves this turn
W_CAPTURE_RISK = 35 # our tokens reachable by enemy eat moves (higher: losing is permanent)
W_FORK = 30 # bonus per eat target beyond the first (tracked separately to avoid double-scaling)
W_EDGE_VULN = 6 # penalty per token on edge/corner (fewer escape routes)
W_CENTRE = 1 # tokens in the central 4×4; tiny so positional never overrides tactics
W_TEMPO = 3 # small constant bonus for the side to move

_BOARD_N = 8
_NCELLS = 64

# Distance from the nearest board edge for each cell
_EDGE_DIST: list[int] = [
    min(r, c, _BOARD_N - 1 - r, _BOARD_N - 1 - c)
    for r in range(_BOARD_N) for c in range(_BOARD_N)
]

# Edge penalty multiplier: 3 on edge/corner, 1 one step in, 0 otherwise
_EDGE_PEN: list[int] = [
    3 if d == 0 else (1 if d == 1 else 0)
    for d in _EDGE_DIST
]

# 1 for cells in rows/cols 2–5, 0 otherwise
_CENTRAL: list[int] = [
    1 if 2 <= r <= 5 and 2 <= c <= 5 else 0
    for r in range(_BOARD_N) for c in range(_BOARD_N)
]

# Precomputed orthogonal neighbours for each cell (flat index)
_NEIGHBOURS: list[list[Coord]] = []
for _r in range(_BOARD_N):
    for _c in range(_BOARD_N):
        _NEIGHBOURS.append([
            Coord(_r + _dr, _c + _dc)
            for _dr, _dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= _r + _dr < _BOARD_N and 0 <= _c + _dc < _BOARD_N
        ])

# Coord to flat index for the tables above
_COORD_TO_IDX: dict[Coord, int] = {
    coord: coord.r * _BOARD_N + coord.c
    for coord in COORD_TABLE
}


def evaluate(board: Board, color: PlayerColor) -> float:
    """Heuristic score from color's perspective; ±INF for terminal states."""
    opp = color.opponent

    my_tokens = board._count_tokens(color)
    opp_tokens = board._count_tokens(opp)

    if opp_tokens == 0:
        return INF
    if my_tokens == 0:
        return -INF

    my_stacks = opp_stacks = 0
    my_centre = opp_centre = 0
    my_edge_pen = opp_edge_pen = 0
    my_threat = opp_threat = 0 # capturable token heights reachable this turn
    my_forks = 0 # eat targets beyond the first (fork bonus)

    # Cache lookups as locals for hot loop speed
    coord_to_idx = _COORD_TO_IDX
    central = _CENTRAL
    edge_pen_tbl = _EDGE_PEN
    neighbours = _NEIGHBOURS

    for coord in COORD_TABLE:
        cell = board[coord]
        if cell.is_empty:
            continue

        idx = coord_to_idx[coord]
        h = cell.height

        if cell.color == color:
            my_stacks += 1
            if central[idx]:
                my_centre += h
            my_edge_pen += edge_pen_tbl[idx] * h

            eat_count = 0
            for nbr in neighbours[idx]:
                nc = board[nbr]
                if nc.is_empty or nc.color == color:
                    continue
                if h >= nc.height:
                    my_threat += nc.height
                    eat_count += 1

            if eat_count > 1:
                my_forks += eat_count - 1 # each extra target is a fork

        else:
            opp_stacks += 1
            if central[idx]:
                opp_centre += h
            opp_edge_pen += edge_pen_tbl[idx] * h

            for nbr in neighbours[idx]:
                nc = board[nbr]
                if nc.is_empty or nc.color == opp:
                    continue
                if h >= nc.height:
                    opp_threat += nc.height

    score = W_MATERIAL * (my_tokens - opp_tokens)
    score += W_STACK * (my_stacks - opp_stacks)
    score += W_CENTRE * (my_centre - opp_centre)
    score += W_CAPTURE_THREAT * my_threat
    score -= W_CAPTURE_RISK * opp_threat
    score += W_FORK * my_forks
    score -= W_EDGE_VULN * (my_edge_pen - opp_edge_pen)
    score += W_TEMPO

    return float(score)