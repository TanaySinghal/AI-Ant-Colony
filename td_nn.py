import numpy as np
import ant_colony as ac
from main import run_game
from greedy import greedy_step
import random
import math
from functools import reduce

import torch
from torch import nn

from td import td_step_fn as good_player_fn

dtype = torch.float
# device = torch.device("cpu")

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

def flatten(arr2d):
  return reduce(lambda x,y : x+y, arr2d)

def utility(board, model):
  # TODO: flatten board
  # nn_input = torch.tensor(flatten(board), dtype=dtype)

  # Do a one hot encoding
  flattened_board = flatten(board)
  nn_input = torch.zeros(64 * 5, dtype=dtype)
  for i in range(len(flattened_board)):
    c = flattened_board[i]
    nn_input[5*i + c] = 1

  # print(nn_input)
  return model(nn_input)

# Return a state
def td_nn_step_fn(model):
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
      new_utility = utility(new_board, model)
      if (best_utility is None) or (new_utility > best_utility):
        best_utility = new_utility
        best_actions = ant_actions
    # if there are no ants left, utility should be very negative
    if len(action_sets) == 0: 
      best_utility = -1
    assert((best_actions is not None) or len(action_sets) == 0)
    return best_utility, best_actions

  def td_step(s, ant_team):
    _, best_actions = get_best_utility(s.board, s.turn_number, ant_team)
    return ac.apply_actions(s, best_actions)

  return td_step

# Takes features, returns weights
def train(ant_team, good_player_step=None, good_player_percent=0, num_games=500, epochs=100, lr=1e-4, gamma=1):
  # No randomness between runs
  torch.manual_seed(0)
  loss_fn = nn.MSELoss()

  def compute_actual_utilities(s_list, gamma):
    N = len(s_list)
    u_list = [0] * N
    for i in range(N-1, -1, -1): # N-1 to 0 [Inclusive, exclusive)
      reward = fitness(s_list[i], ant_team) - fitness(s_list[i-1], ant_team)
      # time_penalty = 5
      if i == N-1:
        u_list[i] = reward
      else:
        u_list[i] = reward + gamma * u_list[i+1]
    return u_list

  # d_input = 64
  # d_h1 = 50
  # d_h2 = 30
  # d_h3 = 15
  d_input = 320
  d_h1 = 200
  d_h2 = 80
  d_h3 = 15
  d_out = 1

  model = nn.Sequential(
    nn.Linear(d_input, d_h1), # TODO: Sigmoid
    nn.ReLU(),
    nn.Linear(d_h1, d_h2),
    nn.ReLU(),
    nn.Linear(d_h2, d_h3),
    nn.ReLU(),
    nn.Linear(d_h3, d_out)
  )

  # model = nn.Sequential(
  #   nn.Linear(d_input, d_h1), # TODO: Sigmoid
  #   nn.ReLU(),
  #   nn.Linear(d_h1, d_h3),
  #   nn.ReLU(),
  #   nn.Linear(d_h3, d_h4),
  #   nn.ReLU(),
  #   nn.Linear(d_h4, d_out)
  # )

  losses = []
  for game_i in range(num_games):
    if good_player_step != None and random.random() < good_player_percent:
      # Run good player
      print("Good player")
      s_list = run_game(good_player_step, greedy_step, True)
    else:
      print("Bad player")
      s_list = run_game(td_nn_step_fn(model), greedy_step, True)

    s_list = s_list[::2] # Only even (red team)

    actual_utilities = torch.tensor(compute_actual_utilities(s_list, gamma), dtype=dtype)
    
    # TODO: Should I go backward?
    # TODO: Loop many epochs over each list
    # print(game_i)
    sum_loss = 0
    for _ in range(epochs):
      sum_loss = 0
      for i in range(len(s_list)):
        u_hat = utility(s_list[i].board, model)[0]
        loss = loss_fn(u_hat, actual_utilities[i])
        sum_loss += loss.item()
        model.zero_grad() # Zero out gradients before running back pass
        loss.backward() # Compute gradients
        with torch.no_grad(): # Upgrade weights using gradient descent
          for w in model.parameters():
            w -= lr * w.grad

    avg_loss = round(sum_loss / len(s_list), 3)
    print("AVG LOSS", game_i, avg_loss)
    print()
    losses.append(avg_loss)
    
    # if game_i % 10 == 0:
    # err = (actual_utilities[0] - utility(s_list[0].board, model))**2 / 2

    # print()
    # print(actual_utilities[0].item(), utility(s_list[0].board, model).item())
    # print("Error:", game_i, round(err.item(), 3), end='\n')

  return model, losses

# theta_blue = [0.48, 0.22, -0.08, -0.02]
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

f_list = [lambda s, ant_team : 1, fitness, sum_dist_to_food, min_dist_to_food]
theta = [0.48, 0.22, -0.08, -0.02]

# TODO: My intuition is that with some exploration we can better ignore random_val
ant_team = ac.Cell.ANT_RED
good_player_step = good_player_fn(theta, f_list)

model, losses = train(ant_team, good_player_step=good_player_step, good_player_percent=1, num_games=300, epochs=50, lr=1e-4, gamma=1)
# Save model
import os
# dir_path = os.path.dirname(os.path.realpath(__file__))
# print(dir_path + "/saved_td_nn")
torch.save(model.state_dict(), os.path.join('saved_model', 'test_model.t7'))

print("--- DONE TRAINING ---")

for name, param in model.named_parameters():
  if param.requires_grad:
      print(name, param.data)

win = 0
trials = 10
saved = []
avg_turns = 0
for t in range(trials):
  print("trial:", t)
  states = run_game(td_nn_step_fn(model), greedy_step, request_history=True)
  end_state = states[len(states)-1]
  if end_state.get_winner() == ac.Cell.ANT_RED:
    win += 1
  avg_turns += end_state.turn_number
  if t == trials-1:
    saved = states
print("Won: %d/%d" % (win, trials))
print("Avg turns ", round(avg_turns/ trials, 3))
#if win > trials * (2/3):
for s in saved:
  print(s)

# Graph loss
print("PLOTTING")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()

plt.title("TD NN Loss")
plt.xlabel("# Games")
plt.ylabel("Loss")
plt.ylim(0,1)
plt.plot(losses)
plt.show()

# Save losses
import numpy
a = numpy.asarray(losses)
numpy.savetxt("losses.csv", a, delimiter=",")