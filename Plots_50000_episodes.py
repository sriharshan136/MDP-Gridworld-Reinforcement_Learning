import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
pi_training = pd.read_csv('data/Results_50K_Iterations/pi_training_50000.csv')
vi_training = pd.read_csv('data/Results_50K_Iterations/vi_training_50000.csv')
mc_training_results = pd.read_csv('data/Results_50K_Iterations/mc_training_results_50000.csv')
q_training_results = pd.read_csv('data/Results_50K_Iterations/q_training_results_50000.csv')
double_q_training_results = pd.read_csv('data/Results_50K_Iterations/double_q_training_results_50000.csv')
triple_q_training_results = pd.read_csv('data/Results_50K_Iterations/triple_q_training_results_50000.csv')


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
fig, axs = plt.subplots(3, 2, figsize=(15, 15))  # Adjust the size as needed

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


# ----------------------------------------------------------------------------------------------

# Function to get the final values from the dataset
def extract_final_values(df):
    final_row = df.iloc[-1]
    return {
        "Episode": final_row['Episode'],
        "Rewards": final_row['Rewards'],
        "Average Rewards": final_row['Average Rewards'],
        "Time Taken": final_row['Time Taken'],
        "Variance": final_row['Variance']
    }


# Get final values from each dataset
final_pi = extract_final_values(pi_training)
final_vi = extract_final_values(vi_training)
final_mc = extract_final_values(mc_training_results)
final_q = extract_final_values(q_training_results)
final_double_q = extract_final_values(double_q_training_results)
final_triple_q = extract_final_values(triple_q_training_results)

# Extract final values
final_values = {
    'Policy Iteration': extract_final_values(pi_training),
    'Value Iteration': extract_final_values(vi_training),
    'Monte Carlo ES': extract_final_values(mc_training_results),
    'Q-Learning': extract_final_values(q_training_results),
    'Double Q-Learning': extract_final_values(double_q_training_results),
    'Triple Q-Learning': extract_final_values(triple_q_training_results)
}

# Convert to DataFrame
final_values_df = pd.DataFrame(final_values).T
# Save to CSV
final_values_df.to_csv('data/CompareData/final_values_50000.csv')

# --------------------------
# Read the final values data
final_values_df = pd.read_csv('data/CompareData/final_values_50000.csv', index_col=0)
final_values_df['Time Taken (seconds)'] = final_values_df['Time Taken'] / 1e9


# Function to create a bar plot
def create_bar_plot(column, title, color):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(final_values_df.index, final_values_df[column],
                   color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow'])
    plt.xlabel('Algorithm')
    plt.ylabel(column)
    plt.title(title)

    plt.xticks(rotation=45)

    # Add the data values on the bars
    for bar in bars:
        y_val = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, y_val, round(y_val, 6),
                 va='bottom')  # Adjust the rounding as needed

    plt.tight_layout()
    plt.show()


# Plot for 'Average Rewards'
create_bar_plot('Average Rewards', 'Average Rewards for Each Algorithm', 'blue')

# Plot for 'Time Taken'
create_bar_plot('Time Taken (seconds)', 'Time Taken by Each Algorithm', 'green')

# Plot for 'Variance'
create_bar_plot('Variance', 'Variance for Each Algorithm', 'red')

# Prepare the data for plotting
datasets = ['PI', 'VI', 'MC', 'Q', 'Double Q', 'Triple Q']
final_values = [final_pi, final_vi, final_mc, final_q, final_double_q, final_triple_q]
metrics = ['Episode', 'Rewards', 'Average Rewards', 'Time Taken', 'Variance']
# print(final_values)

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']

# Plotting separate plots for each metric
for metric in metrics:
    plt.figure(figsize=(15, 10))
    values = [data[metric] for data in final_values]
    plt.bar(datasets, values, color=colors)
    plt.title(f'{metric} for Different Algorithms')
    plt.xlabel('Algorithm')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    for i, value in enumerate(values):
        plt.text(i, value, f'{value}', ha='center', va='bottom')
    plt.show()
