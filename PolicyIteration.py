import matplotlib.pyplot as plt
import numpy as np
import time


class PolicyIteration:
    def __init__(self, world, gamma, init_policy=None, init_value=None):
        """
        Constructor for the PolicyIteration class.

        Parameters:
        - reward_function: 1D array representing the reward for each state.
        - transition_model: 3D array representing the transition probabilities between states and actions.
        - gamma: Discount factor for future rewards.
        - init_policy: Initial policy for the MDP. If None, a random policy is initialized.
        - init_value: Initial state values. If None, values are initialized to zeros.
        """
        # Initialize class attributes
        self.world = world
        self.transition_model = world.transition_model
        self.num_states = world.transition_model.shape[0]
        self.num_actions = world.transition_model.shape[1]
        self.reward_function = np.nan_to_num(world.reward_function)

        self.gamma = gamma

        # Initialize values and policy arrays
        if init_value is None:
            self.values = np.zeros(self.num_states)
        else:
            self.values = init_value

        if init_policy is None:
            self.policy = np.random.randint(0, self.num_actions, self.num_states)
        else:
            self.policy = init_policy

        # self.visualizer = GridWorldVisualizer(self.world, "data/PolicyIteration_output.mp4", "Grid-World (MDP)",
        #                                       "Policy Iteration")

    def test_policy(self):
        total_reward = 0

        # Initialize the state
        state_pos = self.world.start_pos if self.world.start_pos is not None else (7, 0)
        state = self.world.get_state_from_pos(state_pos)

        num_actions = 0
        while True:
            # Choose the action
            action = self.policy[state]

            # Take the action
            s_prime, reward, done = self.world.blackbox_move(state, action)
            total_reward += reward

            num_actions += 1
            if done or num_actions > 100:
                break

            state = s_prime
        return total_reward

    def run_policy_evaluation(self, iteration, tol=0.001):
        """
        Perform policy evaluation for a given policy until convergence or maximum epochs.

        Parameters:
        - tol: Convergence tolerance for policy evaluation.

        Returns:
        - delta_history: Number of iterations performed during policy evaluation.
        """
        epoch = 0
        delta_history = 0
        while epoch < 500:
            delta = 0
            for s in range(self.num_states):
                v = self.values[s]
                a = self.policy[s]
                p = self.transition_model[s, a]
                # Update the state value using the Bellman equation
                self.values[s] = self.reward_function[s] + self.gamma * np.sum(p * self.values)

                delta = max(delta, abs(v - self.values[s]))
            delta_history += 1
            # self.visualizer.plot_policy_values(self.policy, self.values, sub_text=f"Iteration {iteration} - "
            #                                                                       f"Policy Evaluation")
            if delta < tol:
                break
        return delta_history

    def run_policy_improvement(self, iteration):
        """
        Perform policy improvement by selecting actions that maximize expected future rewards.

        Returns:
        - policy_stable: Number of states where the policy remains unchanged.
        """
        policy_stable = 0
        for s in range(self.num_states):
            old_actions = self.policy[s]
            v_list = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                p = self.transition_model[s, a]
                v_list[a] = self.reward_function[s] + self.gamma * np.sum(p * self.values)
            # Update the policy by selecting the action with the maximum expected future reward
            self.policy[s] = np.argmax(v_list)

            # self.visualizer.plot_policy_values(self.policy, self.values, sub_text=f"Iteration {iteration} - "
            #                                                                       f"Policy Improvement")

            if old_actions != self.policy[s]:
                policy_stable += 1
        return policy_stable

    def train(self, iterations=500, tol=0.001, plot=True):
        """
        Train the policy iteration algorithm until convergence or maximum epochs.

        Parameters:
        - tol: Convergence tolerance for policy evaluation.
        - plot: If True, plot the convergence history.
        """
        epoch = 0
        eval_count_history = []
        policy_change_history = []
        reward_history = []
        time_history = [time.perf_counter()]
        not_stable = True

        # self.visualizer.plot_policy_values(self.policy, self.values, sub_text=f"Iteration {epoch}")

        while epoch < iterations:
            epoch += 1
            # Run policy evaluation and improvement steps
            eval_count = self.run_policy_evaluation(iteration=epoch, tol=tol)
            policy_change = self.run_policy_improvement(iteration=epoch)
            if not_stable:
                eval_count_history.append(eval_count)
                policy_change_history.append(policy_change)

            reward_history.append(self.test_policy())
            time_history.append(time_history[-1] + time.perf_counter())
            if policy_change == 0 and not_stable is True:
                not_stable = False
                print(f"Policy became stable by iteration: {epoch}")
                # break

        # Finalize the video after all iterations
        # self.visualizer.finalize_video()

        # Optionally, plot the Final_Animations_Images
        if plot is True:
            fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharex='all', dpi=200)
            axes[0].plot(np.arange(len(eval_count_history)), eval_count_history, marker='o', markersize=4, alpha=0.7,
                         color='#2ca02c', label='# sweeps in \npolicy evaluation\n' + r'$\gamma =$' + f'{self.gamma}')
            axes[0].set_ylabel('Epoch')
            axes[0].set_xlabel('State Values')
            axes[0].legend()

            axes[1].plot(np.arange(len(policy_change_history)), policy_change_history, marker='o', markersize=4,
                         alpha=0.7,
                         color='#d62728',
                         label='# policy updates in \npolicy improvement\n' + r'$\gamma =$' + f'{self.gamma}')
            axes[1].set_xlabel('Policy')
            axes[1].legend()
            fig.text(0.5, 0.02, 'Rate of Change', ha='center')
            plt.tight_layout()
            plt.show()

        return reward_history, time_history
