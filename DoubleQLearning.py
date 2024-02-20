# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import time


# Double Q-Learning class definition
class DoubleQLearning:
    def __init__(self, problem, alpha=0.2, gamma=0.9, epsilon=0.9, xi=0.99):
        """
        Initializes the Q-Learning agent.

        Parameters:
        - problem: The grid world problem to solve.
        - alpha: Learning rate (default is 0.2).
        - gamma: Discount factor for future rewards (default is 0.9).
        - epsilon: Exploration-exploitation trade-off parameter (default is 0.9).
        - xi: Decay factor for epsilon (default is 0.99).
        """
        self.problem = problem
        self.num_states = problem.num_states
        self.num_actions = problem.num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.xi = xi
        self.policy = np.random.randint(self.num_actions, size=self.num_states)
        self.q_table_a = np.zeros((self.num_states, self.num_actions))
        self.q_table_b = np.zeros((self.num_states, self.num_actions))
        self.values = np.zeros(self.num_states)

    def e_greedy(self, s):
        """
        Choose an action based on epsilon-greedy policy.

        Parameters:
        - s: Current state.

        Returns:
        - Action to take.
        """
        if np.random.uniform() <= self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return self.policy[s]

    def update_policy(self, s):
        """
        Update the policy and values based on the current Q-values.

        Parameters:
        - s: Current state.
        """
        self.policy[s] = np.argmax(self.q_table_a[s] + self.q_table_b[s])
        self.values[s] = np.max(self.q_table_a[s] + self.q_table_b[s])

    def train_one_episode(self, start_pos):
        """
        Train the agent for one episode using Q-Learning.

        Parameters:
        - start_pos: Starting position for the episode.

        Returns:
        - reward_game: Total reward obtained in the episode.
        - is_win: Flag indicating whether the episode resulted in a win.
        """
        is_win = 0
        reward_game = 0

        s = self.problem.get_state_from_pos(start_pos)

        while True:
            a = self.e_greedy(s)
            s_prime, r, done = self.problem.blackbox_move(s, a)
            if np.random.uniform() <= 0.5:
                q_prime = np.max(self.q_table_b[s_prime])
                q_value = self.q_table_a[s, a]

                self.q_table_a[s, a] += self.alpha * (r + self.gamma * q_prime - q_value)

            else:
                q_prime = np.max(self.q_table_a[s_prime])
                q_value = self.q_table_b[s, a]

                self.q_table_b[s, a] += self.alpha * (r + self.gamma * q_prime - q_value)

            self.update_policy(s)
            reward_game += r

            if done:
                if r == self.problem.reward[1]:
                    is_win = 1
                break
            else:
                s = s_prime

        return reward_game, is_win

    def train(self, epochs, start_pos, plot=True):
        """
        Train the agent for a specified number of epochs.

        Parameters:
        - epochs: Number of training epochs.
        - start_pos: Starting position for each episode.
        - plot: Boolean indicating whether to plot the training Final_Animations_Images (default is True).
        """
        reward_history = np.zeros(epochs)
        game_win = np.zeros(epochs)
        time_history = [time.perf_counter()]

        for i in range(epochs):
            print(f'Training epoch {i + 1}')
            reward_episode, win_episode = self.train_one_episode(start_pos=start_pos)

            # Decay epsilon for exploration-exploitation trade-off
            self.epsilon *= self.xi

            game_win[i] = win_episode
            reward_history[i] = reward_episode
            time_history.append(time_history[-1] + time.perf_counter())

        print(f'time used = {time_history[-1]}')
        print(f'final reward = {np.sum(reward_history)}')

        if plot:
            # Plot the training Final_Animations_Images
            fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=200, sharex='all')

            # Scatter plot
            axes[0].scatter(np.arange(len(reward_history)), reward_history, marker='o', s=1, alpha=0.7, color='#2ca02c')
            axes[0].set_xlabel('Episode')
            axes[0].set_ylabel('Reward from\na single game')
            axes[0].grid(axis='x')

            # Calculate winning percentage in segments
            segment = 10
            game_win = game_win.reshape((segment, epochs // segment))
            game_win = np.sum(game_win, axis=1)
            print(f'winning percentage = {game_win / (epochs // segment)}')

            axes[1].plot(np.arange(1, segment + 1) * (epochs // segment), game_win / (epochs // segment), marker='o',
                         markersize=2, alpha=0.7, color='#2ca02c')
            axes[1].set_ylabel('Winning percentage')
            axes[1].set_xlabel('Episode')
            axes[1].grid(axis='x')

            plt.tight_layout()
            plt.show()

        return reward_history, time_history
