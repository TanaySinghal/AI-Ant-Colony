import ant_colony as ac
from greedy import greedy_step
import random

MAX_TURNS = 100
RANDOMIZE_FOOD = False

# run_game pits two algorithms against each other and returns state
# step_fn takes in game_state and ant_team and produces new game_state
def run_game(red_step_fn, blue_step_fn, request_history=False):
    if not RANDOMIZE_FOOD:
        random.seed(0)

    game_state = ac.get_init()
    states = []
    if request_history:
        states.append(game_state)

    while (game_state.get_winner() is None) and (game_state.turn_number < MAX_TURNS):
        # print(game_state.to_str())
        # print()
        # print(game_state.turn_number)
        if game_state.get_active_team() == ac.Cell.ANT_RED:
            game_state = red_step_fn(game_state, ac.Cell.ANT_RED)
        else:
            game_state = blue_step_fn(game_state, ac.Cell.ANT_BLUE)
        
        if request_history:
            states.append(game_state)

    # print(game_state.to_str())
    # print()
    return states if request_history else game_state

# Play greedy against itself
# states = run_game(greedy_step, greedy_step)
# print("Winner is ", states[-1].get_winner())
