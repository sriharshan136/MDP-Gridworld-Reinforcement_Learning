# Import necessary libraries and modules
import numpy as np
import pandas as pd
from GridWorld import GridWorld
from ValueIteration import ValueIteration

# Set seed for reproducibility
np.random.seed(100)

# Define the rewards for different states
Reward = {0: -0.02, 1: 1.0, 2: -1.0, 3: np.NaN}

# Create an instance of the GridWorld using a CSV file, reward values, and a random rate
world = GridWorld('data/gridworld.csv', reward=Reward, random_rate=0.3, start_pos=(7, 0))

# Create an instance of the ValueIteration solver with the provided reward function, transition model, and discount
# factor (gamma)
np.random.seed(100)
solver = ValueIteration(world, gamma=0.84)

epoch = 50000

# Train the ValueIteration solver to find the optimal policy and state values
rewards, time_history = solver.train(iterations=epoch)

# Visualize the grid world with state values and the optimal policy obtained from Value Iteration
world.visualize_gridworld(policy=solver.policy, values=solver.values,
                          sub_text='Value Iteration', file_path='data/Final_Animations_Images/ValueIteration_result.png')

# Calculate variance per iteration
variance_per_iteration = np.var(rewards)

# Create a DataFrame for Value Iteration
vi_data = {
    'Episode': np.arange(1, epoch + 1),
    'Time Taken': time_history[1:],
    'Average Rewards': np.cumsum(rewards) / np.arange(1, epoch + 1),
    'Rewards': rewards,
    'Variance': variance_per_iteration / np.arange(1, epoch + 1)
}
vi_df = pd.DataFrame(vi_data)

# Save the DataFrame to a CSV file for Value Iteration
vi_filename = 'data/Results_10K_Iterations/vi_training_results.csv'
vi_df.to_csv(vi_filename, index=False)
