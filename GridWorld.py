import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


class GridWorld:
    def __init__(self, filename, reward, random_rate, start_pos=None):
        """
        Constructor for the GridWorld class.

        Parameters:
        - filename: Name of the CSV file containing the grid map.
        - reward: Array representing the reward for each type of grid cell.
        - random_rate: Probability of a random action in the transition model.
        """
        # Read the map from the given filename
        file = open(filename)
        self.map = np.array(
            [list(map(float, s.strip().split(","))) for s in file.readlines()]
        )
        file.close()

        # Initialize grid world parameters
        self.num_rows = self.map.shape[0]
        self.num_cols = self.map.shape[1]
        self.num_states = self.num_rows * self.num_cols
        self.num_actions = 4
        self.reward = reward
        self.random_rate = random_rate
        self.start_pos = start_pos

        # Calculate reward function and transition model
        self.reward_function = self.get_reward_function()
        self.transition_model = self.get_transition_model()

    def get_state_from_pos(self, pos):
        # Convert grid position to state
        return pos[0] * self.num_cols + pos[1]

    def get_pos_from_state(self, state):
        return state // self.num_cols, state % self.num_cols

    def get_reward_function(self):
        # Calculate the reward function based on the grid map
        reward_table = np.zeros(self.num_states)
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                s = self.get_state_from_pos((r, c))
                reward_table[s] = self.reward[self.map[r, c]]
        return np.nan_to_num(reward_table)

    def get_transition_model(self):
        """
        Calculate the transition model for each state and action.

        Returns:
        - transition_model: 3D array representing the transition model.
        """
        # Calculate the transition model for each state and action
        transition_model = np.zeros((self.num_states, self.num_actions, self.num_states))
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                s = self.get_state_from_pos((r, c))
                neighbor_s = np.zeros(self.num_actions)

                if self.map[r, c] == 0:
                    for a in range(self.num_actions):
                        new_r, new_c = r, c
                        # Calculate the next state based on the action
                        if a == 0:  # Up action
                            new_r = max(r - 1, 0)
                        elif a == 1:  # Right action
                            new_c = min(c + 1, self.num_cols - 1)
                        elif a == 2:  # Down action
                            new_r = min(r + 1, self.num_rows - 1)
                        elif a == 3:  # Left action
                            new_c = max(c - 1, 0)

                        # Check if the action moves to a Wall state
                        if self.map[new_r, new_c] == 3:
                            new_r, new_c = r, c

                        s_prime = self.get_state_from_pos((new_r, new_c))
                        neighbor_s[a] = s_prime
                else:
                    # If the current state is a Wall state, stay in the same state for all actions
                    neighbor_s = np.ones(self.num_actions) * s

                for a in range(self.num_actions):
                    # Update the transition model
                    transition_model[s, a, int(neighbor_s[a])] += 1 - self.random_rate
                    transition_model[s, a, int(neighbor_s[(a + 1) % self.num_actions])] += self.random_rate / 2.0
                    transition_model[s, a, int(neighbor_s[(a - 1) % self.num_actions])] += self.random_rate / 2.0
        return transition_model

    def generate_random_policy(self):
        # Generate a random policy for all states
        return np.random.randint(self.num_actions, size=self.num_states)

    def blackbox_move(self, s, a):
        """
        Simulate a transition in the environment and return the next state and reward.

        Parameters:
        - s: Current state.
        - a: Chosen action.

        Returns:
        - s_prime: Next state after taking action a.
        - r: Reward obtained after taking action a.
        - done: Boolean representing the terminal states.
        """
        # Simulate a transition in the environment and return the next state and reward
        p = self.transition_model[s, a]
        s_prime = np.random.choice(self.num_states, p=p)
        r = self.reward_function[s_prime]
        done = True
        if r == self.reward[0]:
            done = False
        return s_prime, r, done

    def visualize_gridworld(self, policy, values, fig_size=(10, 8), sub_text=None, file_path=None):
        """
        Visualize the grid world with state values and policy.

        Parameters:
        - policy: Optional. 1D array representing the policy for each state.
        - values: 1D array representing the value of each state.
        - fig_size: Tuple representing the size of the visualization figure.
        - sub_text: Optional. Subtitle for the visualization.

        Displays a plot of the grid world with state values and policy arrows.
        """
        # Visualize the grid world with state values and policy
        unit = min(fig_size[1] // self.num_rows, fig_size[0] // self.num_cols)
        unit = max(1, unit)
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax.set_aspect('equal')  # Fix the aspect ratio
        ax.axis('off')

        # Draw grid lines
        for i in range(self.num_cols + 1):
            if i == 0 or i == self.num_cols:
                ax.plot([i * unit, i * unit], [0, self.num_rows * unit],
                        color='black')
            else:
                ax.plot([i * unit, i * unit], [0, self.num_rows * unit],
                        alpha=0.7, color='grey', linestyle='dashed')
        for i in range(self.num_rows + 1):
            if i == 0 or i == self.num_rows:
                ax.plot([0, self.num_cols * unit], [i * unit, i * unit],
                        color='black')
            else:
                ax.plot([0, self.num_cols * unit], [i * unit, i * unit],
                        alpha=0.7, color='grey', linestyle='dashed')

        # Draw grid cells, state values, and policy arrows
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                y = (self.num_rows - 1 - i) * unit
                x = j * unit
                s = self.get_state_from_pos((i, j))

                # Draw walls, goal, and start states
                if self.map[i, j] == 3:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='black')
                    ax.add_patch(rect)
                elif self.map[i, j] == 2:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='red')
                    ax.add_patch(rect)
                elif self.map[i, j] == 1:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='green')
                    ax.add_patch(rect)
                elif self.start_pos is not None:
                    start_s = self.get_state_from_pos(self.start_pos)
                    if start_s == s:
                        rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='yellow')
                        ax.add_patch(rect)

                # Display state values
                if values is not None:
                    if self.map[i, j] != 3:
                        ax.text(x + 0.5 * unit, y + 0.5 * unit, f'{values[s]:.4f}',
                                horizontalalignment='center', verticalalignment='center',
                                fontsize=max(fig_size) * unit * 0.9)

                # Display policy arrows
                if policy is not None:
                    if self.map[i, j] == 0:
                        a = policy[s]
                        symbol = ['^', '>', 'v', '<']
                        ax.plot([x + 0.5 * unit], [y + 0.5 * unit], marker=symbol[a], alpha=0.5,
                                linestyle='none', markersize=max(fig_size) * unit * 1.5, color='#1f77b4')

        # Set the plot title and subtitle
        ax.set_title("Grid-World (MDP): State Values and Policy Visualization")
        fig.text(0.5, 0.05, sub_text, ha='center', fontsize=12)

        # Adjust subplot parameters to control the layout
        plt.subplots_adjust(left=0.0, right=1.0, top=0.94, bottom=0.06)

        if file_path is not None:
            plt.savefig(file_path)
        # Display the plot
        plt.show()
