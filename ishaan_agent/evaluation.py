from referee.game import BOARD_N, Board, PlayerColor, Coord, Direction, GamePhase

WIN_SCORE = 1000000
ALL_DIRECTIONS = (Direction.Up, Direction.Down, Direction.Left, Direction.Right)

def evaluate(board: Board, my_color: PlayerColor) -> float:
    """
    Returns a heuristic score for the board from player's perspective.
    Values tokens, activity, capture chances, and safe board positions.
    """
    if board.game_over:
        winner = board.winner_color
        if winner is None:
            return 0.0
        return WIN_SCORE if winner == my_color else -WIN_SCORE

    opponent = my_color.opponent

    my_tokens = count_tokens(board, my_color)
    opp_tokens = count_tokens(board, opponent)

    my_center = center_score(board, my_color)
    opp_center = center_score(board, opponent)

    my_mobility = pseudo_mobility(board, my_color)
    opp_mobility = pseudo_mobility(board, opponent)

    my_capture = capture_potential(board, my_color)
    opp_capture = capture_potential(board, opponent)

    my_edge_risk = edge_risk(board, my_color)
    opp_edge_risk = edge_risk(board, opponent)

    # reduce importance of centre control outside Placement phase
    if board.phase == GamePhase.PLACEMENT:
        center_weight = 1.5
        mobility_weight = 0.5
    else:
        center_weight = 0.0
        mobility_weight = 1.0

    return (
        12 * (my_tokens - opp_tokens)
        + mobility_weight * (my_mobility - opp_mobility)
        + 6 * (my_capture - opp_capture)
        + 2 * (opp_edge_risk - my_edge_risk)
        + center_weight * (my_center - opp_center)
    )

def count_tokens(board: Board, color: PlayerColor) -> int:
    """
    Counts the total number of tokens a player has on the board.
    Sums the height of all of player's stacks.
    """
    total = 0
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            cell = board[Coord(r, c)]
            if cell.color == color:
                total += cell.height
    return total

def center_score(board: Board, color: PlayerColor) -> float:
    """
    Scores how centrally placed a player's stacks are.
    The closer a stack is to the centre, the better the score, 
    because they are more flexible and harder to be pushed off the board
    """
    score = 0.0
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            cell = board[Coord(r, c)]
            if cell.color == color:
                distance_from_center = abs(r - 3.5) + abs(c - 3.5)
                score -= distance_from_center
    return score

def pseudo_mobility(board: Board, color: PlayerColor) -> int:
    """
    Estimates how many useful actions a player has available.
    Measures based on adjacent Move/Eat options and whether stacks are tall enough to Cascade.
    """
    mobility = 0
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            coord = Coord(r, c)
            cell = board[coord]
            if cell.color != color:
                continue

            for direction in ALL_DIRECTIONS:
                next_r = r + direction.r
                next_c = c + direction.c

                if not (0 <= next_r < BOARD_N and 0 <= next_c < BOARD_N):
                    continue

                target = board[Coord(next_r, next_c)]
                if target.is_empty or target.color == color:
                    mobility += 1
                elif cell.height >= target.height:
                    mobility += 1

    return mobility

def capture_potential(board: Board, color: PlayerColor) -> int:
    """
    Estimates how much a player could capture immediately.
    Rewards positions where adjacent enemy stacks can be eaten.
    """
    potential = 0
    opponent = color.opponent
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            coord = Coord(r, c)
            cell = board[coord]
            if cell.color != color:
                continue

            for direction in ALL_DIRECTIONS:
                next_r = r + direction.r
                next_c = c + direction.c
                if not (0 <= next_r < BOARD_N and 0 <= next_c < BOARD_N):
                    continue

                target = board[Coord(next_r, next_c)]
                if target.color == opponent and cell.height >= target.height:
                    potential += target.height

    return potential

def edge_risk(board: Board, color: PlayerColor) -> float:
    """
    Calculates higher risk if stacks sit closer to the board edge.
    Stacks close to edges are vulnerable to Cascade
    """
    risk = 0.0
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            cell = board[Coord(r, c)]
            if cell.color != color:
                continue

            edge_distance = min(r, c, BOARD_N - 1 - r, BOARD_N - 1 - c)
            if edge_distance == 0:
                risk += 2.0 * cell.height
            elif edge_distance == 1:
                risk += 0.5 * cell.height

    return risk