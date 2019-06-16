import random
from enum import IntEnum

ROW = 8
COL = 8
NUM_ANTS_PER_TEAM = COL
FOOD_PROBABILITY = 0.3 # probability that a tile has food

class Cell(IntEnum):
    EMPTY = 0
    FOOD = 1
    ANT_RED = 2
    ANT_BLUE = 3

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
        team = "Red" if (self.turn_number % 2 == 0) else "Blue"
        board_str = "Round: {} {}'s turn".format(str(self.turn_number),team)
        def cell_to_str(cell):
            if cell == Cell.EMPTY:
                return '·' # NOTE: this is an interpunct and not a period
            elif cell == Cell.FOOD:
                return 'F'
            else:
                return str(int(cell))

        for row in self.board:
            board_str += '\n' + ''.join([cell_to_str(cell) for cell in row])
        return board_str
    
    # returns a winner if only one team remains alive
    def get_winner(self):
        blue_alive = False
        red_alive = False

        for i in range(ROW):
            for j in range(COL):
                if self.board[i][j] == Cell.ANT_BLUE:
                    blue_alive = True
                elif self.board[i][j] == Cell.ANT_RED:
                    red_alive = True
                    
                if blue_alive and red_alive:
                    return None

        return Cell.ANT_RED if red_alive else Cell.ANT_BLUE

# Returns initial game state
def get_init():
    board = [[Cell.EMPTY]*COL for x in [Cell.EMPTY]*ROW]
    for i in range(ROW):
        for j in range(COL):
            num = j + i*COL
            if num < NUM_ANTS_PER_TEAM:
                board[i][j] = Cell.ANT_RED
            elif num >= ROW*COL - NUM_ANTS_PER_TEAM:
                board[i][j] = Cell.ANT_BLUE
            elif random.random() < FOOD_PROBABILITY and i < ROW - 1 and i > 0:
                board[i][j] = Cell.FOOD
    return GameState(board)

# action_list is a list of (i, j, A). Perform action A at coordinate (i,j).
# action_list must perform only one action per ant, 
# and only for the ant team whose turn it is
def apply_actions(game_state, action_list):
    # Determine whose turn it is
    ant_team = (game_state.turn_number % 2) + Cell.ANT_RED

    old_board = game_state.board
    new_board = [row[:] for row in old_board]

    def move_ant(i, j, new_i, new_j):
        if old_board[i][j] != ant_team:
            raise Exception("Attempted to perform action on cell (" + str(i) + "," + str(j) + ") of type " 
            + str(old_board[i][j]) + " but it is " + str(Cell(ant_team)) + "'s turn")

        # If there is food, leave old ant where it was and create new one 
        if old_board[new_i][new_j] != Cell.FOOD:
            new_board[i][j] = Cell.EMPTY

        new_board[new_i][new_j] = old_board[i][j]

    for (i, j, A) in action_list:
        if A == Action.NONE:
            continue
        elif A == Action.LEFT:
            if j > 0:
                move_ant(i, j, i, j-1)
        elif A == Action.UP:
            if i > 0:
                move_ant(i, j, i-1, j)
        elif A == Action.RIGHT:
            if j < COL - 1:
                move_ant(i, j, i, j+1)
        elif A == Action.DOWN:
            if i < ROW - 1:
                move_ant(i, j, i+1, j)
        else:
            raise Exception("Invalid action: ", A)

    return GameState(new_board, game_state.turn_number + 1)

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