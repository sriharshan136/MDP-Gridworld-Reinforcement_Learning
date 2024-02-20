import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval

# Load the datasets
pi_training = pd.read_csv('data/Results_10K_Iterations/pi_training_results.csv')
vi_training = pd.read_csv('data/Results_10K_Iterations/vi_training_results.csv')
mc_training_results = pd.read_csv('data/Results_10K_Iterations/mc_training_results.csv')
q_training_results = pd.read_csv('data/Results_10K_Iterations/q_training_results.csv')
double_q_training_results = pd.read_csv('data/Results_10K_Iterations/double_q_training_results.csv')
triple_q_training_results = pd.read_csv('data/Results_10K_Iterations/triple_q_training_results.csv')


# Function to plot Average Reward Vs Episodes --------------------------------------------------------
def plot_on_subplot(ax, df, title):
    ax.plot(df['Episode'], df['Average Rewards'], label='Average Reward')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    # Annotate the last point
    final_average_reward = df['Average Rewards'].iloc[-1]
    final_episode = df['Episode'].iloc[-1]
    ax.annotate(f'{final_average_reward:.2f}', xy=(final_episode, final_average_reward), textcoords="offset points",
                xytext=(5, 5), ha='center')


# Create subplots
fig, axs = plt.subplots(3, 2, figsize=(15, 10))  # Adjust the size as needed

# Plotting for each training result
plot_on_subplot(axs[0, 0], pi_training, 'Policy Iteration')
plot_on_subplot(axs[0, 1], vi_training, 'Value Iteration')
plot_on_subplot(axs[1, 0], mc_training_results, 'Monte Carlo')
plot_on_subplot(axs[1, 1], q_training_results, 'Q-Learning')
plot_on_subplot(axs[2, 0], double_q_training_results, 'Double Q-Learning')
plot_on_subplot(axs[2, 1], triple_q_training_results, 'Triple Q-Learning')

# Adjust layout
plt.tight_layout()
plt.show()

# Single plot Average Reward Vs Episodes --------------------------------------------------------

# Plot settings
plt.figure(figsize=(15, 10))

# Plot each dataset with the final average reward in the label
plt.plot(pi_training['Episode'], pi_training['Average Rewards'],
         label=f'Policy Iteration (Final: {pi_training["Average Rewards"].iloc[-1]:.2f})')
plt.plot(vi_training['Episode'], vi_training['Average Rewards'],
         label=f'Value Iteration (Final: {vi_training["Average Rewards"].iloc[-1]:.2f})')
plt.plot(mc_training_results['Episode'], mc_training_results['Average Rewards'],
         label=f'Monte Carlo ES (Final: {mc_training_results["Average Rewards"].iloc[-1]:.2f})')
plt.plot(q_training_results['Episode'], q_training_results['Average Rewards'],
         label=f'Q-Learning (Final: {q_training_results["Average Rewards"].iloc[-1]:.2f})')
plt.plot(double_q_training_results['Episode'], double_q_training_results['Average Rewards'],
         label=f'Double Q-Learning (Final: {double_q_training_results["Average Rewards"].iloc[-1]:.2f})')
plt.plot(triple_q_training_results['Episode'], triple_q_training_results['Average Rewards'],
         label=f'Triple Q-Learning (Final: {triple_q_training_results["Average Rewards"].iloc[-1]:.2f})')

# Adding labels and title
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes for Different Learning Methods')
plt.legend()

# Show plot
plt.show()

# Plot all bar plots
# --------------------------
df = pd.read_csv('data/Results_10K_Iterations/data_to_plot.csv')

# Convert time from nanoseconds to seconds
df['Average Time per iteration (seconds)'] = df['Average Time per iteration (nanoseconds)'] / 1e9


# Function to plot bar graph and display values
def plot_bar(column_name, title, y_label):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Algorithms'], df[column_name], color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow'])
    plt.xlabel('Algorithms')
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=45)

    # Add the data values on the bars
    for bar in bars:
        y_val = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, y_val, round(y_val, 6),
                 va='bottom')  # Adjust the rounding as needed

    plt.tight_layout()
    plt.show()


# Plotting Average Reward
plot_bar('Average Reward', 'Average Reward of Algorithms', 'Average Reward')

# Plotting Variance
plot_bar('Variance', 'Variance of Algorithms', 'Variance')

# Plotting Time Taken in Seconds
plot_bar('Average Time per iteration (seconds)', 'Average Time Taken by Algorithms (in seconds)',
         'Average Time per iteration (seconds)')
