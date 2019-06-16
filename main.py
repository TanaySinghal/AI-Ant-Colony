import ant_colony as ac
from greedy import greedy_step
import random

random.seed(0) # Ensure same random numbers every time

# This is our game loop
MAX_ROUND = 100
game_state = ac.get_init()
while game_state.get_winner() is None:
    print(game_state.to_str())
    print()

    if game_state.get_active_team() == ac.Cell.ANT_RED:
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
