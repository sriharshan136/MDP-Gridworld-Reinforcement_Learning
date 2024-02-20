# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import time

from GridWorldVisualizer import GridWorldVisualizer


# Monte Carlo Learning with Exploring Starts (MCLearningES) class definition
class MCLearningES:
    def __init__(self, world, gamma=0.9, ep_length=50):
        """
        Initializes the Monte Carlo Learning with Exploring Starts agent.

        Parameters:
        - problem: The grid world problem to solve.
        - gamma: Discount factor for future rewards (default is 0.9).
        - initial_q: Initial value for the Q-table entries (default is 0.0).
        """
        self.world = world
        self.num_states = world.num_states
        self.num_actions = world.num_actions
        self.gamma = gamma
        self.ep_length = ep_length

        # Q-table to store action values
        self.q_table = np.zeros((self.num_states, self.num_actions))

        # Count table to track the number of visits to state-action pairs
        self.count_table = np.zeros((self.num_states, self.num_actions))

        # Policy table initialized with random actions for each state
        self.policy = np.random.randint(self.num_actions, size=self.num_states)

        # Values table to store the maximum Q-value for each state
        self.values = np.zeros(self.num_states)

        self.visited_policy = np.full(self.num_states, None, dtype=object)
        self.visited_values = np.full(self.num_states, None, dtype=object)

        # self.visualizer = GridWorldVisualizer(self.world, "data/Final_Animations_Images/MCLearningES_output.mp4", "Grid-World (MDP)",
        #                                       "Monte Carlo with Exploring Starts")

    def policy_update(self, s):
        """
        Update the policy of State s based on its current Q-values.
        """
        self.policy[s] = np.argmax(self.q_table[s])
        self.values[s] = np.max(self.q_table[s])

        self.visited_policy[s] = self.policy[s]
        self.visited_values[s] = self.values[s]

    def train_one_episode(self, iteration):
        """
        Train the agent for one episode using Monte Carlo Exploring Starts.

        Returns:
        - reward_game: Total reward obtained in the episode.
        - is_win: Flag indicating whether the episode resulted in a win.
        """
        is_win = 0
        reward_game = 0
        time_start = int(round(time.time() * 1000))

        g = 0  # Initialize return
        s = np.random.randint(self.num_states)  # Choose a random starting state
        a = np.random.randint(self.num_actions)  # Choose a random starting action
        episode = []  # Store the state-action-reward tuples for the episode

        # Main loop for the episode
        while True:
            s_prime, r, done = self.world.blackbox_move(s, a)
            episode.append((s, a, r))

            s = s_prime
            a = self.policy[s]  # Choose action according to current policy
            reward_game += r

            # Check if the agent is in terminal state
            if done:
                if r == self.world.reward[1]:
                    is_win = 1
                break

            # Length check for the episode
            if len(episode) > self.ep_length:
                break

            # Time limit check for the episode
            if int(round(time.time() * 1000)) - time_start > 1000:
                print('Time out in this episode!')
                break

        # Backward update of Q-values for the episode
        for idx, step in enumerate(episode[::-1]):
            g = self.gamma * g + step[2]

            # First visit check
            # if step[0] not in np.array(episode[::-1])[:, 0][idx + 1:]:

            # Every Visit.
            self.count_table[step[0], step[1]] += 1
            self.q_table[step[0], step[1]] += (g - self.q_table[step[0], step[1]]) \
                                              / self.count_table[step[0], step[1]]

            # Update the policy based on the updated Q-table
            self.policy_update(step[0])

            # if iteration < 100:
            #     self.visualizer.plot_policy_values(self.visited_policy, self.visited_values,
            #                                        sub_text=f"Iteration {iteration}")

        return reward_game, is_win

    def train(self, epochs, plot=True):
        """
        Train the agent for a specified number of epochs.

        Parameters:
        - epochs: Number of training epochs.
        - plot: Boolean indicating whether to plot the training Final_Animations_Images (default is True).
        """
        reward_history = np.zeros(epochs)
        game_win = np.zeros(epochs)
        time_history = [time.perf_counter()]
        # self.visualizer.plot_policy_values(self.visited_policy, self.visited_values, sub_text=f"Iteration {0}")
        # Main training loop
        for i in range(epochs):

            # if i % 100 == 0:
            #     print(f'Training epoch {i}')

            # Train one episode and record Final_Animations_Images
            reward_episode, win_episode = self.train_one_episode(iteration=i)
            # self.visualizer.plot_policy_values(self.visited_policy, self.visited_values,
            #                                    sub_text=f"Iteration {i}")
            game_win[i] = win_episode
            reward_history[i] = reward_episode
            time_history.append(time_history[-1] + time.perf_counter())

        # Finalize the video after all iterations
        # self.visualizer.finalize_video()

        # print(f'time used = {time_history[-1]}')
        # print(f'Average reward = {np.mean(reward_history)}')

        # Plot the training Final_Animations_Images
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
