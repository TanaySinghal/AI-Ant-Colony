import random
from enum import IntEnum

ROW, COL = 8, 8
NUM_ANTS_PER_TEAM = 2
FOOD_PROBABILITY = 0.15 # 0.15 probability that a tile has food
AMOUNT_OF_FOOD = 4

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
    
    def __str__(self):
        team = "Red" if (self.get_active_team() == Cell.ANT_RED) else "Blue"
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
    
    # determine which team's turn it is
    def get_active_team(self):
        return Cell.ANT_RED if (self.turn_number % 2 == 0) else Cell.ANT_BLUE

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
    food_added = 0
    for i in range(ROW):
        for j in range(COL):
            num = j + i*COL
            if num < NUM_ANTS_PER_TEAM:
                board[i][j] = Cell.ANT_RED
            elif num >= ROW*COL - NUM_ANTS_PER_TEAM:
                board[i][j] = Cell.ANT_BLUE
            elif random.random() < FOOD_PROBABILITY and 0 < i < ROW/2 and food_added < AMOUNT_OF_FOOD :
                board[i][j] = Cell.FOOD
                food_added += 1
            elif i >= ROW/2 and board[ROW-i-1][j] == Cell.FOOD:
                board[i][COL-j-1] = Cell.FOOD
                food_added += 1
    return GameState(board)

# Returns new ant coordinates, if valid
def move_ant(i, j, board, action):
    old_board = board
    new_i, new_j = action_to_coord(action, (i, j))

    if not inside_board((new_i, new_j)):
        return None
    if old_board[i][j] != Cell.ANT_BLUE and old_board[i][j] != Cell.ANT_RED:
        raise Exception("Attempted to perform action on cell (" + str(i) + "," + str(j) + ") of type " 
        + str(old_board[i][j]) + ".")

    return new_i, new_j

def apply_action_to_board(board, action_list):
    ant_team = board[action_list[0][0]][action_list[0][1]]

    new_board = [row[:] for row in board]

    for i in range(ROW):
        for j in range(COL):
            if board[i][j] == ant_team:
                new_board[i][j] = Cell.EMPTY
    
    for (i, j, action) in action_list:
        new_i, new_j = action_to_coord(action, (i, j))

        if not inside_board((new_i, new_j)):
            raise Exception("Cannot move ant outside board")

        # If ant moves to food spawn food at team baseline
        # If there is no space at baseline no new ant is spawned
        if board[new_i][new_j] == Cell.FOOD:
            if ant_team == Cell.ANT_RED:
                for j in range(COL):
                    if new_board[0][j] != ant_team:
                        new_board[0][j] = ant_team
                        break
            else:
                for j in range(COL):
                    if new_board[ROW-1][COL-1-j] != ant_team:
                        new_board[ROW-1][COL-1-j] = ant_team
                        break
        new_board[new_i][new_j] = ant_team
    return new_board

# action_list is a list of (i, j, A). Perform action A at coordinate (i,j).
# action_list must perform only one action per ant, 
# and only for the ant team whose turn it is
def apply_actions(state, action_list):
    new_board = apply_action_to_board(state.board, action_list)
    return GameState(new_board, state.turn_number + 1)

# Returns all possible actions for ants in ant_team
# as a list of ant actions
def get_actions(board, ant_team):
    # Get actions for specific ant
    def get_actions_for_ant(i, j):
        ant_actions = []
        for action in range(len(Action)):
            if move_ant(i, j, board, Action(action)) is not None:
                ant_actions.append((i, j, Action(action)))
        return ant_actions

    ant_actions_list = []
    for i in range(ROW):
        for j in range(COL):
            if board[i][j] == ant_team:
                ant_actions_list.append(get_actions_for_ant(i, j))
    return all_possible_action_sets(ant_actions_list)

def all_possible_action_sets(ant_actions_list):
  if len(ant_actions_list) == 0:
    return [[]]

  # head is possible actions for ant 1
  # tail is rest of ant_actions_list
  possible_actions_ant1, *rest = ant_actions_list

  # Gives all possible combinations without ant 1
  rest_combinations = all_possible_action_sets(rest)

  c = []
  for action_ant1 in possible_actions_ant1:
    # Add this action for ant 1 to each permutation
    new_perm = []
    for permutation in rest_combinations:
      new_perm.append([action_ant1] + permutation)
    c = c + new_perm
  return c

def dist(coord1, coord2):
    return abs(coord1[1]-coord2[1]) + abs(coord1[0]-coord2[0])

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