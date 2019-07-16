import ant_colony as ac
import random

MAX_TURNS = 100
RANDOMIZE_FOOD = True
COUNTER = 0     # Only relevant if not randomizing food
COUNTER_MAX = 5 # Only relevant if not randomizing food

# run_game pits two algorithms against each other and returns state
# step_fn takes in game_state and ant_team and produces new game_state
def run_game(red_step_fn, blue_step_fn, request_history=False):
    if not RANDOMIZE_FOOD:
        global COUNTER
        random.seed(COUNTER)
        COUNTER = (COUNTER + 1) % COUNTER_MAX

    game_state = ac.get_init()
    states = []
    if request_history:
        states.append(game_state)

    while (game_state.get_winner() is None) and (game_state.turn_number < MAX_TURNS):
        if game_state.get_active_team() == ac.Cell.ANT_RED:
            game_state = red_step_fn(game_state, ac.Cell.ANT_RED)
        else:
            game_state = blue_step_fn(game_state, ac.Cell.ANT_BLUE)
        
        if request_history:
            states.append(game_state)

    return states if request_history else game_state
