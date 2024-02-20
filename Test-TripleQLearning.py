# Import necessary libraries and modules
import numpy as np
import pandas as pd

from GridWorld import GridWorld
from TripleQLearning import TripleQLearning

# Set random seed for reproducibility
np.random.seed(100)

# Define reward values for the GridWorld problem
Reward = {0: -0.02, 1: 1.0, 2: -1.0, 3: np.NaN}

# Create a GridWorld problem instance
problem = GridWorld('data/gridworld.csv', reward=Reward, random_rate=0.3, start_pos=(7, 0))

# Initialize a Triple QLearner with specified parameters
np.random.seed(100)
solver_triple = TripleQLearning(problem, alpha=0.35, gamma=0.8, epsilon=0.85, xi=0.99)

# Number of training epochs
epoch = 10000

# Train the Q-learning model and plot the training Final_Animations_Images
rewards, time_history = solver_triple.train(epoch, start_pos=(7, 0), plot=True)

# Visualize the learned policy and values on the GridWorld
problem.visualize_gridworld(policy=solver_triple.policy, values=solver_triple.values, sub_text='Triple Q-Learning',
                            file_path='data/Final_Animations_Images/TQLearning_result.png')

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
filename = 'data/Results_10K_Iterations/triple_q_training_results.csv'
df.to_csv(filename, index=False)
