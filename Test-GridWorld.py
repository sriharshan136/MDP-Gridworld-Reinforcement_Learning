# Import necessary libraries and modules
import numpy as np
import pandas as pd
from GridWorld import GridWorld
from GridWorldVisualizer import GridWorldVisualizer

# Set seed for reproducibility
np.random.seed(100)

# Define the reward values for different states
Reward = {0: -0.02, 1: 1.0, 2: -1.0, 3: np.NaN}

# Create an instance of the GridWorld using a CSV file, reward values, and a random rate
world = GridWorld('data/gridworld.csv', reward=Reward, random_rate=0.3, start_pos=(7, 0))

# Generate a random policy for the initial visualization
init_policy = world.generate_random_policy()

# Visualize the initial policy with state values set to zero and a specified subtitle
world.visualize_gridworld(init_policy, np.zeros(world.num_states), sub_text="Random Policy")

# Print the reward function for each state
reward_function = world.reward_function
print('Reward function:')
for s in range(len(reward_function)):
    print(f'State s = {s}, Reward R({s}) = {reward_function[s]}')

# Print the transition model probabilities for each state-action pair and next state
transition_model = world.transition_model
print('Transition model:')
for s in range(transition_model.shape[0]):
    print('======================================')
    for a in range(transition_model.shape[1]):
        print('--------------------------------------')
        for s_prime in range(transition_model.shape[2]):
            print(f's = {s}, a = {a}, s\' = {s_prime}, p = {transition_model[s, a, s_prime]}')
