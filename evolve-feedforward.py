from __future__ import print_function
import os
import neat
import visualize

from functools import reduce

import ant_colony as ac
from main import run_game, RANDOMIZE_FOOD

GENERATIONS = 300 # How many generations to keep running

# Input is flattened list of board state (one-hot encode the possible numbers). MxNx4 and 2x1 for curr ant
# Output is a 5D vector for each action

def argmax(arr):
    best_index = 0
    largest = arr[0]
    for i in range(len(arr)):
        if arr[i] > largest:
            best_index = i
            largest = arr[i]
            
    return best_index

# Returns a function
def neural_net_step_fn(neural_net):
    def to_one_hot(val):
        arr = [0]*4
        arr[val] = 1
        return arr

    def flatten(arr2d):
        return reduce(lambda x,y :x+y, arr2d)

    def neural_net_step(game_state, ant_team):
        board = game_state.board
        actions = []
        for i in range(ac.ROW):
            for j in range(ac.COL):
                if board[i][j] == ant_team:
                    # One-hot encoding
                    # one_hot_arr = flatten([to_one_hot(x) for x in flatten(board)])
                    # output = neural_net.activate([i / ac.ROW, j / ac.COL] + one_hot_arr)
                    output = neural_net.activate([i / ac.ROW, j / ac.COL] + flatten(board))
                    action = argmax(output)
                    actions.append((i, j, action))
        
        return ac.apply_actions(game_state, actions)

    return neural_net_step

def get_fitness_food(net):
    neural_net_step = neural_net_step_fn(net)
    fitness = 0
    food_coords = set()
    N = 1 if RANDOMIZE_FOOD else 1
    for _ in range(N):
        end_game_state = run_game(neural_net_step)
        board = end_game_state.board
        for i in range(ac.ROW):
            for j in range(ac.COL):
                if board[i][j] == ac.Cell.ANT_RED:
                    ant_coord = (i,j)
                elif board[i][j] == ac.Cell.FOOD:
                    food_coords.add((i,j))
    if len(food_coords) == 0:
        return 1
    distance = 0
    for food_coord in food_coords:
        distance += ac.dist(ant_coord, food_coord)
    return -1*distance

# For each genome, calculate fitness
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = get_fitness_food(net)

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval_genomes, GENERATIONS)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    print("Best fitness: ", get_fitness_food(winner_net))

    # Run a sample game with this net
    print("Sample game: \n")
    neural_net_step = neural_net_step_fn(winner_net)
    states = run_game(neural_net_step, True)
    counter = 0
    for s in states:
        print(s.to_str())
        print()

    # node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    # visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
