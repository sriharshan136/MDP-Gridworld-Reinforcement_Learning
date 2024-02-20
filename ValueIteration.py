# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import time

from GridWorldVisualizer import GridWorldVisualizer


class ValueIteration:
    def __init__(self, world, gamma):
        """
        Constructor for the ValueIteration class.

        Parameters:
        - reward_function: 1D array representing the reward for each state.
        - transition_model: 3D array representing the transition probabilities between states and actions.
        - gamma: Discount factor for future rewards.
        """
        # Initialize class attributes
        self.world = world
        self.num_states = world.transition_model.shape[0]
        self.num_actions = world.transition_model.shape[1]

        # Replace NaN with 0 in the reward function (handling missing or undefined rewards)
        self.reward_function = np.nan_to_num(world.reward_function)
        self.transition_model = world.transition_model
        self.gamma = gamma

        # Initialize state values to zero
        self.values = np.zeros(self.num_states)
        self.policy = None  # Placeholder for the optimal policy

        # self.visualizer = GridWorldVisualizer(self.world, "data/ValueIteration_output.mp4", "Grid-World (MDP)",
        #                                       "Value Iteration")

    def test_policy(self):
        total_reward = 0

        # Initialize the state
        state_pos = self.world.start_pos if self.world.start_pos is not None else (7, 0)
        state = self.world.get_state_from_pos(state_pos)

        num_actions = 0
        while True:
            # Choose the action
            v_list = np.zeros(self.num_actions)  # Initialize a list to store expected future rewards for each action
            for a in range(self.num_actions):
                p = self.transition_model[state, a]  # Transition probabilities for the current state-action pair

                # Calculate the expected future reward for each action
                v_list[a] = self.reward_function[state] + self.gamma * np.sum(p * self.values)

            # Choose the action that maximizes the expected future reward
            action = np.argmax(v_list)

            # Take the action
            s_prime, reward, done = self.world.blackbox_move(state, action)
            total_reward += reward

            num_actions += 1
            if done or num_actions > 100:
                break

            state = s_prime
        return total_reward

    def run_iteration(self, iteration):
        """
        Perform a single iteration of the value iteration algorithm.

        Returns:
        - delta: The maximum change in state values during this iteration.
        """
        delta = 0  # Initialize the maximum change to zero
        for s in range(self.num_states):
            v = self.values[s]  # Current value of the state
            v_list = np.zeros(self.num_actions)  # Initialize a list to store expected future rewards for each action
            for a in range(self.num_actions):
                p = self.transition_model[s, a]  # Transition probabilities for the current state-action pair

                # Calculate the expected future reward for each action
                v_list[a] = self.reward_function[s] + self.gamma * np.sum(p * self.values)

            # Update the state value to the maximum expected future reward
            self.values[s] = max(v_list)

            # Update the maximum change if the change in value for this state is greater
            delta = max(delta, abs(v - self.values[s]))

        # self.visualizer.plot_policy_values(self.policy, self.values, sub_text=f"Iteration {iteration} - "
        #                                                                       f"Value Evaluation")
        return delta

    def get_policy(self):
        """
        Determine the optimal policy based on the current state values.

        Returns:
        - pi: 1D array representing the optimal action for each state.
        """
        pi = np.ones(self.num_states) * -1  # Initialize policy to -1 (placeholder)
        for s in range(self.num_states):
            v_list = np.zeros(self.num_actions)  # Initialize a list to store expected future rewards for each action
            for a in range(self.num_actions):
                p = self.transition_model[s, a]  # Transition probabilities for the current state-action pair

                # Calculate the expected future reward for each action
                v_list[a] = self.reward_function[s] + self.gamma * np.sum(p * self.values)

            # Choose the action that maximizes the expected future reward
            pi[s] = np.argmax(v_list)

        return pi.astype(int)

    def train(self, iterations=500, tol=0.001, plot=True):
        """
        Train the value iteration algorithm until convergence.

        Parameters:
        - tol: Convergence tolerance (stop when the maximum change in state values is below this threshold).
        - plot: If True, plot the convergence history.
        """
        epoch = 0
        delta_history = []  # Track the maximum change in state values over iterations
        reward_history = []
        time_history = [time.perf_counter()]
        not_converge = True

        # self.visualizer.polt_policy_values(self.policy, self.values, sub_text=f"Iteration {epoch}")

        while epoch < iterations:  # Limit the number of iterations to avoid infinite loops
            epoch += 1
            delta = self.run_iteration(iteration=epoch)
            if not_converge:
                delta_history.append(delta)

            reward_history.append(self.test_policy())
            time_history.append(time_history[-1] + time.perf_counter())
            if delta < tol and not_converge is True:
                not_converge = False
                # print(f"Policy converged by iteration: {epoch}")
                # Stop if the algorithm has converged
                # break

        self.policy = self.get_policy()  # Set the optimal policy based on the final state values

        # self.visualizer.plot_policy_values(self.policy, self.values, sub_text="Updating Policy")

        # Finalize the video after all iterations
        # self.visualizer.finalize_video()

        if plot is True:
            # Plot the convergence history
            fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=200)
            ax.plot(np.arange(len(delta_history)) + 1, delta_history, marker='o', markersize=4,
                    alpha=0.7, color='#2ca02c', label=r'$\gamma= $' + f'{self.gamma}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Delta')
            ax.legend()
            plt.tight_layout()
            plt.show()

        return reward_history, time_history
