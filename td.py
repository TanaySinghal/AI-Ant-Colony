import numpy as np
import ant_colony as ac
from main import run_game
from greedy import greedy_step
import random
import math
from functools import reduce

def fitness(s):
  board = s.board
  ant_team = s.get_active_team()
  assert(ant_team == ac.Cell.ANT_RED)
  net_ants = 0
  for i in range(ac.ROW):
      for j in range(ac.COL):
          if board[i][j] == ac.Cell.ANT_RED:
            net_ants += 1
          elif board[i][j] == ac.Cell.ANT_BLUE:
            net_ants -= 1

  return net_ants

# Return a state
def td_step_fn(theta, f_list, training = True):
  def td_step(s, ant_team):
    action_list = ac.get_actions(s.board, ant_team)
    best_action_list = []
    board = s.board
    for ant_actions in action_list:
      best_utility = None
      best_action = None
      best_board = None
      for action in ant_actions:
        new_board = ac.apply_action_to_board(board, [action])
        new_utility = utility(ac.GameState(new_board, s.turn_number), theta, f_list)
        if (best_utility is None) or (new_utility > best_utility):
          best_utility = new_utility
          best_action = action
          best_board = new_board
      assert(best_action is not None)
      best_action_list.append(best_action)
      board = best_board
    
    return ac.apply_actions(s, best_action_list)
  return td_step

def utility(s, theta, f_list):
  f_applied = np.array(list(map(lambda f: f(s), f_list)))
  return np.dot(theta, f_applied)

# Takes features, returns weights
def train(f_list, epochs, lr, gamma = 1):
  M = len(f_list)

  def update_theta(s, actual_utility, theta, lr):
    theta_new = np.zeros(M)
    expected_utility = utility(s, theta, f_list)
    for i in range(M):
      theta_new[i] = theta[i] + lr * (actual_utility - expected_utility) * (f_list[i])(s)
    return theta_new

  # TODO: Try using geometric series
  def compute_actual_utilities(s_list, gamma):
    N = len(s_list)
    u_list = [0] * N
    # TODO: Idea: instead, of computing delta fitness, 
    # just compute fitness at a particular state
    for i in range(N-1, -1, -1): # N-1 to 0 [Inclusive, exclusive)
      reward = fitness(s_list[i]) - fitness(s_list[i-1])
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
      err = (actual_utilities[0] - utility(s_list[0], theta, f_list))**2 / 2
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
def sum_dist_to_food(s):
  board = s.board
  ant_team = s.get_active_team()
  assert(ant_team == ac.Cell.ANT_RED)
  
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

def min_dist_to_food(s):
  board = s.board
  ant_team = s.get_active_team()
  assert(ant_team == ac.Cell.ANT_RED)
  
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

def near_enemy(s):
  board = s.board
  ant_team = s.get_active_team()
  assert(ant_team == ac.Cell.ANT_RED)
  for i in range(ac.ROW):
      for j in range(ac.COL):
          if board[i][j] == ant_team:
            look_for = ac.Cell.ANT_BLUE
            coords = [(i+1,j+1),(i+1,j-1),(i-1,j+1), (i-1,j-1)]
            for coord in coords:
              if ac.inside_board(coord) and board[coord[0]][coord[1]] == look_for:
                return 1
  return 0

def near_friend(s):
  board = s.board
  ant_team = s.get_active_team()
  assert(ant_team == ac.Cell.ANT_RED)
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

def near_food(s):
  board = s.board
  ant_team = s.get_active_team()
  assert(ant_team == ac.Cell.ANT_RED)
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

def num_friends(s):
  board = s.board
  ant_team = s.get_active_team()
  assert(ant_team == ac.Cell.ANT_RED)
  net_ants = 0
  for i in range(ac.ROW):
      for j in range(ac.COL):
          if board[i][j] == ac.Cell.ANT_RED:
            net_ants += 1
  return net_ants

def num_enemies(s):
  board = s.board
  ant_team = s.get_active_team()
  assert(ant_team == ac.Cell.ANT_RED)
  net_ants = 0
  for i in range(ac.ROW):
      for j in range(ac.COL):
          if board[i][j] == ac.Cell.ANT_BLUE:
            net_ants += 1
  return net_ants

def num_turns(s):
  return s.turn_number

# If we use this as a feature,
# it should have 0 weight
def random_val(s):
  return random.random()

# TODO: My intuition is that with some exploration we can better ignore random_val
f_list = [lambda s : 1, fitness, sum_dist_to_food, min_dist_to_food]
# f_list = [lambda s : 1, fitness, num_friends, num_enemies, sum_dist_to_food, min_dist_to_food, num_turns]
theta = train(f_list, 500, 1e-4, .9)

print("--- DONE TRAINING ---")
print("THETA:", end=' ')
for t in theta:
  print(round(t, 2), end=' ')
print()

win = 0
trials = 100
saved = []
avg_turns = 0
for t in range(trials):
  states = run_game(td_step_fn(theta, f_list, False), greedy_step, True)
  end_state = states[len(states)-1]
  if end_state.get_winner() == ac.Cell.ANT_RED:
    win += 1
  avg_turns += end_state.turn_number
  if t == trials-1:
    saved = states
print("Won: %d/%d" % (win, trials))
print("Avg turns ", round(avg_turns/ trials, 3))
if win > trials * (2/3):
  for s in saved:
    print(s)
