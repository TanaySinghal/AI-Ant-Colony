import random
from enum import IntEnum

ROW, COL = 12, 8
NUM_ANTS_PER_TEAM = 1
AMOUNT_OF_FOOD = 5 # amount of food on board

class Cell(IntEnum):
    EMPTY = 0
    FOOD = 1
    ANT_RED = 2

class Action(IntEnum):
    NONE = 0
    LEFT = 1
    UP = 2
    RIGHT = 3
    DOWN = 4

# NOTE: this is a pure data class
# It should NOT contain member functions that mutate its properties
class GameState:
    def __init__(self, board, turn_number=0):
        self.board = board # Board is 2D array of Cell
        self.turn_number = turn_number
    
    def to_str(self):
        team = "Red"
        board_str = "Round: {} {}'s turn".format(str(self.turn_number),team)
        def cell_to_str(cell):
            if cell == Cell.EMPTY:
                return '·' # NOTE: this is an interpunct and not a period
            elif cell == Cell.FOOD:
                return 'x'
            elif cell == Cell.ANT_RED:
                return 'R'
            else:
                return 'B'

        for row in self.board:
            board_str += '\n' + ''.join([cell_to_str(cell) for cell in row])
        return board_str

    def food_left(self):
        for i in range(ROW):
            for j in range(COL):
                if self.board[i][j] == Cell.FOOD:
                    return True
        return False

    def food_locations(self):
        food_locations = []
        for i in range(ROW):
            for j in range(COL):
                if self.board[i][j] == Cell.FOOD:
                    food_locations.append([i,j])
        return food_locations

# Returns initial game state
def get_init():
    board = [[Cell.EMPTY]*COL for x in [Cell.EMPTY]*ROW]
    food_positions = set()
    while  len(food_positions) < AMOUNT_OF_FOOD:
        random_food_row = random.randrange(1, ROW)
        random_food_col = random.randrange(1, COL)
        food_positions.add((random_food_row, random_food_col))
    for position in food_positions:
        board[position[0]][position[1]] = Cell.FOOD
    for i in range(ROW):
        for j in range(COL):
            num = j + i*COL
            if num < NUM_ANTS_PER_TEAM:
                board[i][j] = Cell.ANT_RED
    return GameState(board)

# action_list is a list of (i, j, A). Perform action A at coordinate (i,j).
# action_list must perform only one action per ant, 
# and only for the ant team whose turn it is
def apply_actions(game_state, action_list):
    old_board = game_state.board
    new_board = [row[:] for row in old_board]

    def move_ant(i, j, new_i, new_j):
        if i == j and new_i == new_j:
            return
        if not inside_board((new_i, new_j)):
            return

        # If ant moves to food spawn food at team baseline
        # If there is no space at baseline no new ant is spawned
        if old_board[new_i][new_j] == Cell.FOOD:
            for j in range(COL):
                if new_board[0][j] == Cell.EMPTY:
                    new_board[0][j] = Cell.ANT_RED
                    break

        new_board[i][j] = Cell.EMPTY
        new_board[new_i][new_j] = old_board[i][j]

    for (i, j, action) in action_list:
        next_coord = action_to_coord(action, (i, j))
        move_ant(i, j, next_coord[0], next_coord[1])
        
    return GameState(new_board, game_state.turn_number + 1)

def dist(coord1, coord2):
    return (abs(coord1[1]-coord2[1]) + abs(coord1[0]-coord2[0]))

def closest(coord, locations):
    return None if not locations else min(locations,key=lambda x:dist(coord,x))

def inside_board(coord):
    return (0 <= coord[0] < ROW and 0 <= coord[1] < COL)
    
def action_to_coord(action, pos=(0,0)):
    if action == Action.NONE:
        return pos
    elif action == Action.LEFT:
        return (pos[0], pos[1]-1)
    elif action == Action.RIGHT:
        return (pos[0], pos[1]+1)
    elif action == Action.UP:
        return (pos[0]-1, pos[1])
    elif action == Action.DOWN:
        return (pos[0]+1, pos[1])
    
    raise Exception("Invalid action: ", action)

def coord_to_action(from_coord, to_coord):
    if from_coord[1]-to_coord[1] > 0:
        return Action.LEFT
    elif from_coord[1]-to_coord[1] < 0:
        return Action.RIGHT
    elif from_coord[0]-to_coord[0] > 0:
        return Action.UP
    elif from_coord[0]-to_coord[0] < 0:
        return Action.DOWN
    return Action.NONE

# This is how the state works
# game_state = get_init()
# print(game_state.to_str())
# print()

# red_actions = [(0, 0, Action.RIGHT), (0, 3, Action.DOWN)]
# game_state = apply_actions(game_state, red_actions)
# print(game_state.to_str())
# print()

# blue_actions = [(ROW-1, 0, Action.RIGHT), (ROW-1, 2, Action.UP)]
# game_state = apply_actions(game_state, blue_actions)
# print(game_state.to_str())