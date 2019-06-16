import ant_colony as ac
import random

random.seed(0) # Ensure same random numbers every time

# Perform one greedy step and return new game state (deep copy)
# Greedy step rules:
# 1. Every ant will head towards the closest piece of food to them
# 2. If an enemy is 1 move away and food is 1 move away, attack enemy first
# 3. Avoid enemy as much as possible while still advancing towards food
# (maximize distance to enemy, minimize distance to food)
# Ex. |1|-|-|     |-|1|-|
#     |-|F|-| ->  |-|F|-|
#     |2|-|-|     |2|-|-|
# 4. When no more food left, ants head towards the closest enemy ant
def greedy_step(game_state, ant_team):
    board = game_state.board
    enemy = ac.Cell.ANT_BLUE if ant_team == ac.Cell.ANT_RED else ac.Cell.ANT_RED

    # Find food and enemies to optimize step
    food_locations = []
    enemies = []
    for i in range(ac.ROW):
        for j in range(ac.COL):
            if board[i][j] == ac.Cell.FOOD:
                food_locations.append((i,j))
            elif board[i][j] == enemy:
                enemies.append((i,j))

    def dist(coord1, coord2):
        return (abs(coord1[1]-coord2[1]) + abs(coord1[0]-coord2[0]))

    def closest(coord1,locations):
        return None if not locations else min(locations,key=lambda x:dist(coord1,x))

    def get_action(at_coord, to_coord):
        if at_coord[1]-to_coord[1] > 0:
            return ac.Action.LEFT
        elif at_coord[1]-to_coord[1] < 0:
            return ac.Action.RIGHT
        elif at_coord[0]-to_coord[0] > 0:
            return ac.Action.UP
        elif at_coord[0]-to_coord[0] < 0:
            return ac.Action.DOWN
        return ac.Action.NONE

    # Get all steps possible from the at_coord
    # Choose step that minimizes min_coord (food) and maximizes max_coord (enemy)
    def min_risk_max_reward_step(at_coord, min_coord, max_coord, board, teammate_positions):
        def min_max_dist(at_coord):
            if min_coord == None: # if there is no food left just attack
                return dist(at_coord, max_coord)
            else:
                return (abs(dist(at_coord, min_coord))*10 - abs(dist(at_coord, max_coord)))

        all_steps = [
            (at_coord[0], at_coord[1]),
            (at_coord[0], at_coord[1] + 1),
            (at_coord[0], at_coord[1] - 1),
            (at_coord[0] + 1, at_coord[1]),
            (at_coord[0] - 1, at_coord[1])]

        def inside_board(step):
            return (0 <= step[0] < ac.ROW and 0 <= step[1] < ac.COL)

        def criss_cross(step):
            piece_moved = step[0] < at_coord[0] or step[1] < at_coord[1]
            ally_located_at_step = board[step[0]][step[1]] == ant_team
            return (at_coord in teammate_positions and piece_moved and ally_located_at_step)

        # remove steps that kill teammate ants (remove friendly fire) and off board steps
        # Also prevents criss-cross steps (when ant1 chooses to move to ant2's position
        # and ant2 chooses to move to ant1's position)
        possible_steps = [step for step in all_steps if (step not in teammate_positions
                            and inside_board(step) and not criss_cross(step))]
        return min(possible_steps, key=lambda x:min_max_dist(x))

    # Define actions per ant
    actions = []
    teammate_positions = set()
    for i in range(ac.ROW):
        for j in range(ac.COL):
            if board[i][j] == ant_team:
                closest_food = closest((i, j), food_locations)
                closest_enemy = closest((i, j), enemies)
                if dist((i,j), closest_enemy) == 1 and closest_enemy not in teammate_positions:
                    step = closest_enemy
                else:
                    step = min_risk_max_reward_step((i,j), closest_food, closest_enemy, board, teammate_positions)
                action = get_action((i,j), step)
                actions.append((i, j, action))
                teammate_positions.add(step)

    # Orders action list to allow ants to move safely
    # Ants moving to an empty/enemy/food position will move first
    def order_actions(actions):
        def new_pos(pos, action):
            if action == ac.Action.LEFT:
                return (pos[0], pos[1]-1)
            elif action == ac.Action.RIGHT:
                return (pos[0], pos[1]+1)
            elif action == ac.Action.UP:
                return (pos[0]-1, pos[1])
            elif action == ac.Action.DOWN:
                return (pos[0]+1, pos[1])
            else:
                return pos
        def safe_action(new_position, board):
            return board[new_position[0]][new_position[1]] != ant_team
        new_board = [row[:] for row in board]
        ordered_actions = []
        while actions:
            for action in actions:
                pos = (action[0], action[1])
                act = action[2]
                new_position = new_pos(pos, act)
                safe = safe_action(new_position, new_board) or pos == new_position
                if safe:
                    new_board[new_position[0]][new_position[1]] = ant_team
                    new_board[pos[0]][pos[1]] = ac.Cell.EMPTY
                    actions.remove(action)
                    ordered_actions.append(action)
        return ordered_actions

    return ac.apply_actions(game_state, order_actions(actions))

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