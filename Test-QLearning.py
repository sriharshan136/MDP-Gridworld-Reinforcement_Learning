# Import necessary libraries and modules
import numpy as np
import pandas as pd

from GridWorld import GridWorld
from QLearning import QLearning

# Set random seed for reproducibility
np.random.seed(100)

# Define reward values for the GridWorld problem
Reward = {0: -0.02, 1: 1.0, 2: -1.0, 3: np.NaN}

# Create a GridWorld problem instance
world = GridWorld('data/gridworld.csv', reward=Reward, random_rate=0.3, start_pos=(7, 0))

np.random.seed(100)
# Initialize a QLearner with specified parameters
solver = QLearning(world,gamma=0.99, alpha=0.3, epsilon=0.9, xi=0.99)

# Number of training epochs
epoch = 10000

# Train the Q-learning model and plot the training Final_Animations_Images
rewards, time_history = solver.train(epoch, start_pos=(7, 0), plot=True)

# Visualize the learned policy and values on the GridWorld
world.visualize_gridworld(policy=solver.policy, values=solver.values, sub_text='Q - Learning',
                          file_path='data/Final_Animations_Images/QLearning_result.png')

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
filename = 'data/Results_10K_Iterations/q_training_results.csv'
df.to_csv(filename, index=False)
