# Import necessary libraries and modules
import numpy as np
import pandas as pd
from GridWorld import GridWorld
from PolicyIteration import PolicyIteration

# Set seed for reproducibility
np.random.seed(100)

# Define the reward values for different states
Reward = {0: -0.02, 1: 1.0, 2: -1.0, 3: np.NaN}

# Create an instance of the GridWorld using a CSV file, reward values, and a random rate
world = GridWorld('data/gridworld.csv', reward=Reward, random_rate=0.3, start_pos=(7, 0))

# Generate a random policy for the initial state
init_policy = world.generate_random_policy()

# Create an instance of the PolicyIteration solver with the provided reward function, transition model, and discount
# factor
np.random.seed(100)
solver = PolicyIteration(world, gamma=0.9, init_policy=init_policy)

epoch = 100

# Train the PolicyIteration solver to find the optimal policy and state values
rewards, time_history = solver.train(iterations=epoch, tol=0.01)

# Visualize the grid world with state values and the optimal policy obtained from Value Iteration
world.visualize_gridworld(policy=solver.policy, values=solver.values,
                          sub_text='Policy Iteration', file_path='data/Final_Animations_Images/PolicyIteration_result.png')

# Calculate variance per iteration
variance_per_iteration = np.var(rewards)

# Create a DataFrame for Policy Iteration
pi_data = {
    'Episode': np.arange(1, epoch+1),
    'Time Taken': time_history[1:],
    'Average Rewards': np.cumsum(rewards) / np.arange(1, epoch+1),
    'Rewards': rewards,
    'Variance': variance_per_iteration / np.arange(1, epoch + 1)
}
pi_df = pd.DataFrame(pi_data)

# Save the DataFrame to a CSV file for Policy Iteration
pi_filename = 'data/Results_10K_Iterations/pi_training_results.csv'
pi_df.to_csv(pi_filename, index=False)
