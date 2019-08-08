import numpy as np
import ant_colony as ac
from main import run_game
from greedy import greedy_step
import random
import math
from functools import reduce

def fitness(s, ant_team):
  board = s.board
  net_ants = 0
  opponent = ac.Cell.ANT_RED if ant_team == ac.Cell.ANT_BLUE else ac.Cell.ANT_BLUE
  for i in range(ac.ROW):
    for j in range(ac.COL):
      if board[i][j] == ant_team:
        net_ants += 1
      elif board[i][j] == opponent:
        net_ants -= 1

  return net_ants

def opponent(ant_team):
  return ac.Cell.ANT_RED if (ant_team == ac.Cell.ANT_BLUE) else ac.Cell.ANT_BLUE

# Return a state
def td_step_fn(theta, f_list, minmax = False):
  # Find best utility not considering minimizing opponents moves 
  def get_best_utility(board, turn_number, ant_team):
    action_sets = ac.get_actions(board, ant_team)
    # if len(action_sets) > 50000:
    #     s = ac.GameState(board, 1)
    #     print(s)
    #     print("I'm taking a long time since I'm looking through", len(action_sets), "possible action sets")
    best_utility = None
    best_actions = None
    for ant_actions in action_sets:
      new_board = ac.apply_action_to_board(board, ant_actions)
      new_utility = utility(ac.GameState(new_board, turn_number), ant_team, theta, f_list)
      if (best_utility is None) or (new_utility > best_utility):
        best_utility = new_utility
        best_actions = ant_actions
    # if there are no ants left, utility should be very negative
    if len(action_sets) == 0: best_utility = -1
    assert((best_actions is not None) or len(action_sets) == 0)
    return best_utility, best_actions

  def td_step(s, ant_team):
    if not minmax:
      best_utility, best_actions = get_best_utility(s.board, s.turn_number, ant_team)
      return ac.apply_actions(s, best_actions)
    else:
      action_sets = ac.get_actions(s.board, ant_team)
      if len(action_sets) > 5000:
          print("I'm taking a long time since I'm looking through", len(action_sets), "possible action sets")
      best_utility = None
      best_actions = None
      for ant_actions in action_sets: 
        new_board = ac.apply_action_to_board(s.board, ant_actions)
        max_utility = utility(ac.GameState(new_board, s.turn_number), ant_team, theta, f_list)
        enemy_best_utility, enemy_best_actions = get_best_utility(new_board, s.turn_number+1, opponent(ant_team))
        assert(num_enemies(new_board, ant_team) == 0 or enemy_best_utility != None)
        new_utility = max_utility - enemy_best_utility
        if (best_utility is None) or (new_utility > best_utility):
          best_utility = new_utility
          best_actions = ant_actions
      return ac.apply_actions(s, best_actions)
  return td_step

def utility(s, ant_team, theta, f_list):
  f_applied = np.array(list(map(lambda f: f(s, ant_team), f_list)))
  return np.dot(theta, f_applied)

# Takes features, returns weights
def train(f_list, ant_team, epochs, lr, gamma = 1):
  M = len(f_list)

  def update_theta(s, actual_utility, theta, lr):
    theta_new = np.zeros(M)
    expected_utility = utility(s, ant_team, theta, f_list)
    for i in range(M):
      theta_new[i] = theta[i] + lr * (actual_utility - expected_utility) * (f_list[i])(s, ant_team)
    return theta_new

  # TODO: Try using geometric series
  def compute_actual_utilities(s_list, gamma):
    N = len(s_list)
    u_list = [0] * N
    # TODO: Idea: instead, of computing delta fitness, 
    # just compute fitness at a particular state
    for i in range(N-1, -1, -1): # N-1 to 0 [Inclusive, exclusive)
      reward = fitness(s_list[i], ant_team) - fitness(s_list[i-1], ant_team)
      if i == N-1:
        u_list[i] = reward
      else:
        u_list[i] = reward + gamma * u_list[i+1]
    return u_list

  theta = (np.random.rand(M) - 0.5)*0.1

  for epoch in range(epochs):
    s_list = run_game(td_step_fn(theta, f_list), greedy_step, True)
    s_list = s_list[::2] # Only even (red team)
    actual_utilities = compute_actual_utilities(s_list, gamma)
    for i in range(len(s_list)):
      theta = update_theta(s_list[i], actual_utilities[i], theta, lr)
    
    if epoch % 10 == 0:
      err = (actual_utilities[0] - utility(s_list[0], ant_team, theta, f_list))**2 / 2
      print(epoch, round(err, 3), end='\t')
      print("THETA:", end=' ')
      for t in theta:
        print(round(t, 3), end=' ')
      print()

  return theta

# Helper
def get_foods(board):
  food_list = []
  for i in range(ac.ROW):
    for j in range(ac.COL):
      if board[i][j] == ac.Cell.FOOD:
        food_list.append((i,j))
  return food_list

# FEATURE FUNCTIONS
def sum_dist_to_food(s, ant_team):
  board = s.board
  food_list = get_foods(board)
  sum_dist = 0
  num_ants = 0
  for i in range(ac.ROW):
    for j in range(ac.COL):
        if board[i][j] == ant_team:
          num_ants += 1
          for f in food_list:
            sum_dist = sum_dist + ac.dist(f, (i,j)) / ac.ROW
  if num_ants > 0:
    return sum_dist / num_ants
  return sum_dist

def min_dist_to_food(s, ant_team):
  board = s.board
  food_list = get_foods(board)
  min_dist = None
  for i in range(ac.ROW):
    for j in range(ac.COL):
      if board[i][j] == ant_team:
        for f in food_list:
          if min_dist is None:
            min_dist = ac.dist(f, (i,j)) / ac.ROW
          else:
            min_dist = min(min_dist, ac.dist(f, (i,j))) / ac.ROW
  return 0 if min_dist is None else min_dist

def near_enemy(s, ant_team):
  board = s.board
  for i in range(ac.ROW):
    for j in range(ac.COL):
      if board[i][j] == ant_team:
        look_for = ac.Cell.ANT_RED if (ant_team == ac.Cell.ANT_BLUE) else ac.Cell.ANT_BLUE
        coords = [(i+1,j+1),(i+1,j-1),(i-1,j+1), (i-1,j-1)]
        for coord in coords:
          if ac.inside_board(coord) and board[coord[0]][coord[1]] == look_for:
            return 1
  return 0

def near_friend(s, ant_team):
  board = s.board
  ret_val = 0
  for i in range(ac.ROW):
    for j in range(ac.COL):
      if board[i][j] == ant_team:
        look_for = ant_team
        coords = [(i+1,j+1),(i+1,j-1),(i-1,j+1), (i-1,j-1)]
        for coord in coords:
          if ac.inside_board(coord) and board[coord[0]][coord[1]] == look_for:
            ret_val += 1
  return ret_val

def near_food(s, ant_team):
  board = s.board
  ret_val = 0
  for i in range(ac.ROW):
    for j in range(ac.COL):
      if board[i][j] == ant_team:
        look_for = ac.Cell.FOOD
        coords = [(i+1,j+1),(i+1,j-1),(i-1,j+1), (i-1,j-1)]
        for coord in coords:
          if ac.inside_board(coord) and board[coord[0]][coord[1]] == look_for:
            ret_val += 1
  return ret_val

def num_friends(s, ant_team):
  board = s.board
  net_ants = 0
  for i in range(ac.ROW):
    for j in range(ac.COL):
      if board[i][j] == ant_team:
        net_ants += 1
  return net_ants

def num_enemies(board, ant_team):
  net_ants = 0
  for i in range(ac.ROW):
    for j in range(ac.COL):
      if board[i][j] == opponent(ant_team):
        net_ants += 1
  return net_ants

def num_turns(s):
  return s.turn_number

# If we use this as a feature,
# it should have 0 weight
def random_val(s):
  return random.random()

# TODO: My intuition is that with some exploration we can better ignore random_val
# f_list = [lambda s, ant_team : 1, fitness, sum_dist_to_food, min_dist_to_food]
# ant_team = ac.Cell.ANT_RED
# epoch = 500
# # theta = train(f_list, ant_team, epoch, 1e-4, .9)
# # theta = [0.24, 0.06, -0.03, 0.01]
# theta = train(f_list, ant_team, epoch, 1e-4, .9)

# print("--- DONE TRAINING ---")
# print("THETA:", end=' ')
# for t in theta:
#   print(round(t, 2), end=' ')
# print()

# win = 0
# trials = 100
# saved = []
# avg_turns = 0
# for t in range(trials):
#   print("trial:", t)
#   # run the min max version of TD against the only-max version of TD
#   # td_step_fn(theta, f_list, True)
#   states = run_game(td_step_fn(theta, f_list), greedy_step, True)
#   end_state = states[len(states)-1]
#   if end_state.get_winner() == ac.Cell.ANT_RED:
#     win += 1
#   avg_turns += end_state.turn_number
#   if t == trials-1:
#     saved = states
# print("Won: %d/%d" % (win, trials))
# print("Avg turns ", round(avg_turns/ trials, 3))
# #if win > trials * (2/3):
# for s in saved:
#   print(s)
