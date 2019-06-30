import ant_colony as ac
from greedy import greedy_step
import random

MAX_TURNS = 100
RANDOMIZE_FOOD = True

# run_game pits two algorithms against each other and returns state
# step_fn takes in game_state and ant_team and produces new game_state
def run_game(red_step_fn, request_history=False):
    if not RANDOMIZE_FOOD:
        random.seed(0)

    game_state = ac.get_init()
    states = []
    if request_history:
        states.append(game_state)

    while game_state.food_left() and (game_state.turn_number < MAX_TURNS):
        # print(game_state.to_str())
        # print()
        # print(game_state.turn_number)
        game_state = red_step_fn(game_state, ac.Cell.ANT_RED)

        if request_history:
            states.append(game_state)

    # print(game_state.to_str())
    # print()
    return states if request_history else game_state

# Play greedy against itself
# states = run_game(greedy_step, greedy_step)
# print("Winner is ", states[-1].get_winner())
