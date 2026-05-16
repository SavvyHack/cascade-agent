from referee.game import (BOARD_N, Board, Coord, Direction, GamePhase, PlayerColor, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction, IllegalActionException)
from math import inf
from .evaluation import evaluate, count_tokens

ALL_DIRECTIONS = (Direction.Up, Direction.Down, Direction.Left, Direction.Right)

def choose_action(board: Board, my_color: PlayerColor, max_depth: int) -> Action:
    """
    Returns the legal action with the best minimax score.
    Each action is applied, then recursively searched deeper (next moves) using minimax.
    """
    legal_actions = order_actions(generate_legal_actions(board))
    # before running minimax, check whether there is a move that wins or removes opponent tokens right away
    if board.phase == GamePhase.PLAY:
        tactical_action = choose_tactical_action(board, my_color, legal_actions)
        if tactical_action is not None:
            return tactical_action

    best_action = legal_actions[0]
    best_score = -inf

    alpha = -inf
    beta = inf

    # test score of each legal action by evaluating the action by doing then undoing it
    for action in legal_actions:
        board.apply_action(action)
        score = minimax(board, max_depth - 1, my_color, alpha, beta)
        board.undo_action()

        if score > best_score:
            best_score = score
            best_action = action
        elif score == best_score and action_priority(action) > action_priority(best_action):
            best_action = action
        
        alpha = max(alpha, best_score)

    return best_action

def choose_tactical_action(board: Board, my_color: PlayerColor, legal_actions: list[Action]) -> Action | None:
    """
    Checks for an immediate tactical action before doing full minimax.
    Prefers moves that immediately win or remove the most opponent tokens.
    """
    opponent = my_color.opponent
    opp_tokens_before = count_tokens(board, opponent)
    best_action = None
    best_gain = 0

    for action in legal_actions:
        board.apply_action(action)
        if board.game_over and board.winner_color == my_color:
            board.undo_action()
            return action

        opp_tokens_after = count_tokens(board, opponent)
        token_gain = opp_tokens_before - opp_tokens_after
        board.undo_action()
        if token_gain > best_gain:
            best_gain = token_gain
            best_action = action

    return best_action

def minimax(board: Board, depth: int, my_color: PlayerColor, alpha: float, beta: float) -> float:
    """
    Returns the minimax score of the current board for player using alpha-beta pruning.

    Returns the evaluation score at base case. Otherwise, searches legal actions 
    and assumes each player chooses moves in their own best interest. Skips branches
    that cannot improve the final result.
    """
    # base case: returns evaluation score if search depth limit is reached or game is over
    if depth == 0 or board.game_over:
        return evaluate(board, my_color)

    # make sure there are legal actions
    legal_actions = order_actions(generate_legal_actions(board))
    if not legal_actions:
        return evaluate(board, my_color)

    # player's turn - maximise: choose move with highest score
    if board.turn_color == my_color:
        score = -inf
        for action in legal_actions:
            board.apply_action(action)
            score = max(score, minimax(board, depth - 1, my_color, alpha, beta))
            board.undo_action()

            # stop exploring branch if it can no longer affect the final decision
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        return score

    # opponent's turn - minimise: choose move with lowest score
    score = inf
    for action in legal_actions:
        board.apply_action(action)
        score = min(score, minimax(board, depth - 1, my_color, alpha, beta))
        board.undo_action()

        # stop exploring branch if it can no longer affect the final decision
        beta = min(beta, score)
        if alpha >= beta:
            break

    return score

def generate_legal_actions(board: Board) -> list[Action]:
    """
    Generate all legal actions for the current player on the given board.
    """
    actions: list[Action] = []

    # if in Placement phase, test whether each square is a legal Place action
    if board.phase == GamePhase.PLACEMENT:
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                action = PlaceAction(Coord(r, c))
                if is_legal_action(board, action):
                    actions.append(action)
        return actions

    current_color = board.turn_color

    # if in Play phase, test whether each action for each stack of current player's colour is a legal action
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            coord = Coord(r, c)
            cell = board[coord]

            if cell.color != current_color:
                continue

            for direction in ALL_DIRECTIONS:
                candidate_actions = (
                    MoveAction(coord, direction),
                    EatAction(coord, direction),
                    CascadeAction(coord, direction),
                )

                for action in candidate_actions:
                    # keep only legal actions
                    if is_legal_action(board, action):
                        actions.append(action)
    return actions

def is_legal_action(board: Board, action: Action) -> bool:
    """
    Check whether an action is legal on the current board.
    Attempts to apply the action, undoing if succesful.
    """
    try:
        board.apply_action(action)
        board.undo_action()
        return True
    except IllegalActionException:
        return False

def action_priority(action: Action) -> int:
    """
    Rank actions for move ordering and tie-breaking.
    """
    if isinstance(action, EatAction):
        return 3
    if isinstance(action, MoveAction):
        return 2
    if isinstance(action, CascadeAction):
        return 1
    return 0

def order_actions(actions: list[Action]) -> list[Action]:
    """
    Return actions ordered from highest to lowest priority.
    """
    return sorted(actions, key=action_priority, reverse=True)