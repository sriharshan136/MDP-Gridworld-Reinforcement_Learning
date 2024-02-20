import numpy as np
import pandas as pd

from GridWorld import GridWorld
from MCLearningES import MCLearningES
from PolicyIteration import PolicyIteration
from QLearning import QLearning
from ValueIteration import ValueIteration


def p_parameters():
    data = {
        'Episode': [],
        'Gamma': [],
        'Tolerances': [],
        'Initial Policy': [],
        'Average time': [],
        'Average reward': []
    }

    gamma_val = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    tol_val = [0.1, 0.05, 0.01, 0.005, 0.001]
    init_policy = [np.zeros(64, dtype=int), np.ones(64, dtype=int) * 1, np.ones(64, dtype=int) * 2,
                   np.ones(64, dtype=int) * 3, np.random.randint(0, 4, 64)]
    episodes = [100, 500, 1000, 5000, 10000]
    i = 0
    for gamma in gamma_val:
        for tol in tol_val:
            for idx, policy in enumerate(init_policy):
                i += 1
                print(f"Progress: {i}")
                for ep in episodes:
                    # Set seed for reproducibility
                    np.random.seed(100)
                    solver = PolicyIteration(world, gamma=gamma, init_policy=policy)
                    rewards, time_history = solver.train(iterations=ep, tol=tol, plot=False)
                    avg_reward = np.mean(rewards)
                    avg_time = time_history[-1] / ep

                    data['Episode'].append(ep)
                    data['Gamma'].append(gamma)
                    data['Tolerances'].append(tol)
                    data['Initial Policy'].append(idx)  # 4 means Random Policy
                    data['Average time'].append(avg_time)
                    data['Average reward'].append(avg_reward)

    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    filename = 'data/Hyper_Parameters/p_parameters.csv'
    df.to_csv(filename, index=False)


def v_parameters():
    data = {
        'Episode': [],
        'Gamma': [],
        'Average time': [],
        'Average reward': []
    }

    gamma_val = np.arange(0.7, 1.0, 0.01)
    episodes = [100, 500, 1000, 2000, 5000, 10000]
    i = 0
    for gamma in gamma_val:
        i += 1
        print(f"Progress: {i}")
        for ep in episodes:
            # Set seed for reproducibility
            np.random.seed(100)
            solver = ValueIteration(world, gamma=gamma)
            rewards, time_history = solver.train(iterations=ep, plot=False)
            avg_reward = np.mean(rewards)
            avg_time = time_history[-1] / ep

            data['Episode'].append(ep)
            data['Gamma'].append(gamma)
            data['Average time'].append(avg_time)
            data['Average reward'].append(avg_reward)

    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    filename = 'data/Hyper_Parameters/v_parameters.csv'
    df.to_csv(filename, index=False)


def mc_parameters():
    data = {
        'Episode': [],
        'Gamma': [],
        'Episode Length': [],
        'Average time': [],
        'Average reward': []
    }

    gamma_val = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    ep_lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    episodes = [1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]

    i = 0
    for gamma in gamma_val:
        for ep_length in ep_lengths:

            i += 1
            print(f"Progress: {i}")
            for ep in episodes:
                # Set seed for reproducibility
                np.random.seed(100)
                solver = MCLearningES(world, gamma=gamma, ep_length=ep_length)
                rewards, time_history = solver.train(ep, plot=False)
                avg_reward = np.mean(rewards)
                avg_time = time_history[-1] / ep

                data['Episode'].append(ep)
                data['Gamma'].append(gamma)
                data['Episode Length'].append(ep_length)
                data['Average time'].append(avg_time)
                data['Average reward'].append(avg_reward)

    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    filename = 'data/Hyper_Parameters/mc_parameters.csv'
    df.to_csv(filename, index=False)


def q_parameters():
    data = {
        'Episode': [],
        'Gamma': [],
        'Alpha': [],
        'Epsilon': [],
        'Average time': [],
        'Average reward': []
    }

    alpha_val = [0.1, 0.15, 0.20, 0.25, 0.3, 0.35]
    gamma_val = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    epsilon_val = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    episodes = [100, 500, 1000, 2000, 5000, 8000, 9000, 10000]

    for gamma in gamma_val:
        for alpha in alpha_val:
            for epsilon in epsilon_val:
                for ep in episodes:
                    # Set seed for reproducibility
                    np.random.seed(100)
                    solver = QLearning(world, alpha=alpha, gamma=gamma, epsilon=epsilon)
                    rewards, time_history = solver.train(ep, start_pos=(7, 0), plot=False)
                    avg_reward = np.mean(rewards)
                    avg_time = time_history[-1] / ep

                    data['Episode'].append(ep)
                    data['Gamma'].append(gamma)
                    data['Alpha'].append(alpha)
                    data['Epsilon'].append(epsilon)
                    data['Average time'].append(avg_time)
                    data['Average reward'].append(avg_reward)

    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    filename = 'data/Hyper_Parameters/q_parameters.csv'
    df.to_csv(filename, index=False)


# Set seed for reproducibility
np.random.seed(100)

# Define the reward values for different states in the grid world
Reward = {0: -0.02, 1: 1.0, 2: -1.0, 3: np.NaN}

# Create an instance of the GridWorld using a CSV file, reward values, and a random rate
world = GridWorld('data/gridworld.csv', reward=Reward, random_rate=0.3, start_pos=(7, 0))

# Uncomment the below method calls to run parameter check.
# v_parameters()
# p_parameters()
# q_parameters()
# mc_parameters()
