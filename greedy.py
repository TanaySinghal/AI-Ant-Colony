import ant_colony as ac
import random

random.seed(0) # Ensure same random numbers every time

# Perform one greedy step and return new game state (deep copy)
def greedy_step(game_state, ant_team):
    board = game_state.board
    
    # Create actions
    actions = []
    
    # TODO: Fix this (I am doing random action for each ant)
    # NOTE: An invalid action will appear to do nothing (e.g. moving ant out of board)
    for i in range(ac.ROW):
        for j in range(ac.COL):
            if board[i][j] == ant_team:
                rand_action = random.randint(0, len(ac.Action)-1)
                actions.append((i, j, ac.Action(rand_action)))

    # Apply actions
    return ac.apply_actions(game_state, actions)

# This is basically our "game loop"
MAX_ROUND = 100
game_state = ac.get_init()
while game_state.get_winner() is None:
    print(game_state.to_str())
    print()

    is_red_turn = (game_state.turn_number % 2) == 0
    if is_red_turn:
        # Use greedy strategy for red team
        game_state = greedy_step(game_state, ac.Cell.ANT_RED)
    else:
        # NOTE: We would use a different algo for blue but greedy is all we have for now
        game_state = greedy_step(game_state, ac.Cell.ANT_BLUE)
    
    # NOTE: We wouldn't use this normally, but for development purposes
    # let's avoid an infinite loop
    if game_state.turn_number >= MAX_ROUND:
        break

print(game_state.to_str())
print()
print("Winner is ", game_state.get_winner())