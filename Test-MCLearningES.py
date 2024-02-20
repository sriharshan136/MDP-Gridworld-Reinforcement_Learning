# Import necessary libraries and modules
import numpy as np
import pandas as pd
from GridWorld import GridWorld
from MCLearningES import MCLearningES

# Set seed for reproducibility
np.random.seed(100)

# Define the reward values for different states in the grid world
Reward = {0: -0.02, 1: 1.0, 2: -1.0, 3: np.NaN}

# Create an instance of the GridWorld using a CSV file, reward values, and a random rate
world = GridWorld('data/gridworld.csv', reward=Reward, random_rate=0.3, start_pos=(7, 0))

# Create an instance of the MCLearnerES class for the defined grid world with initial Q-values set to 0.0
np.random.seed(100)
solver = MCLearningES(world, gamma=0.85, ep_length=80)

# Set the number of training epochs
epoch = 50000

# Train the MCLearningES agent for the specified number of epochs and plot the Final_Animations_Images
rewards, time_history = solver.train(epoch, plot=True)

# Visualize the grid world with state values and the optimal policy obtained from Monte Carlo Exploring Starts
world.visualize_gridworld(policy=solver.policy, values=solver.values, sub_text='Monte Carlo Exploring Starts',
                          file_path='data/Final_Animations_Images/MCLearningES_result.png')

# Calculate variance per iteration
variance_per_iteration = np.var(rewards)

# Create a DataFrame for Q-Learning Final_Animations_Images
data = {
    'Episode': np.arange(1, epoch + 1),
    'Rewards': rewards,
    'Average Rewards': np.cumsum(rewards) / np.arange(1, epoch+1),
    'Time Taken': time_history[1:],
    'Variance': variance_per_iteration / np.arange(1, epoch + 1)
}
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
filename = 'data/Results_10K_Iterations/mc_training_results.csv'
df.to_csv(filename, index=False)