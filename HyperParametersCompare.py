import numpy as np
import pandas as pd

q_parameters = pd.read_csv('data/Hyper_Parameters/q_parameters.csv')
dq_parameters = pd.read_csv('data/Hyper_Parameters/dq_parameters.csv')
mc_parameters = pd.read_csv('data/Hyper_Parameters/mc_parameters.csv')
tq_parameters = pd.read_csv('data/Hyper_Parameters/tq_parameters.csv')
p_parameters = pd.read_csv('data/Hyper_Parameters/p_parameters.csv')
v_parameters = pd.read_csv('data/Hyper_Parameters/v_parameters.csv')

# Assuming you want to maximize average reward and minimize average time
# First, sort by highest average reward and then by lowest average time
sorted_q_parameters = q_parameters.sort_values(by=['Average reward', 'Average time'], ascending=[False, True])
sorted_dq_parameters = dq_parameters.sort_values(by=['Average reward', 'Average time'], ascending=[False, True])
sorted_mc_parameters = mc_parameters.sort_values(by=['Average reward', 'Average time'], ascending=[False, True])
sorted_tq_parameters = tq_parameters.sort_values(by=['Average reward', 'Average time'], ascending=[False, True])
sorted_p_parameters = p_parameters.sort_values(by=['Average reward', 'Average time'], ascending=[False, True])
sorted_v_parameters = v_parameters.sort_values(by=['Average reward', 'Average time'], ascending=[False, True])

# Get the top row (optimal values)
q_parameters_optimal_values = sorted_q_parameters.iloc[0]
dq_parameters_optimal_values = sorted_dq_parameters.iloc[0]
mc_parameters_optimal_values = sorted_mc_parameters.iloc[0]
tq_parameters_optimal_values = sorted_tq_parameters.iloc[0]
p_parameters_optimal_values = sorted_p_parameters.iloc[0]
v_parameters_optimal_values = sorted_v_parameters.iloc[0]

# Print the optimal values
print("Optimal Parameters of Value Iteration:")
#Episode,Gamma,Average time,Average reward
print("Gamma:", v_parameters_optimal_values['Gamma'])
print("Episode:", v_parameters_optimal_values['Episode'])
print("Average Reward:", v_parameters_optimal_values['Average reward'])
print("Average Time:", v_parameters_optimal_values['Average time'])
print('\n')
print('\n')

print("Optimal Parameters of Policy iteration:")
#Episode,Gamma,Tolerances,Initial Policy,Average time,Average reward
print("Gamma:", p_parameters_optimal_values['Gamma'])
print("Episode:", p_parameters_optimal_values['Episode'])
print("Tolerances:", p_parameters_optimal_values['Tolerances'])
print("Initial Policy:", p_parameters_optimal_values['Initial Policy'])
print("Average Reward:", p_parameters_optimal_values['Average reward'])
print("Average Time:", p_parameters_optimal_values['Average time'])
print('\n')
print('\n')


print("Optimal Parameters of MC:")
#Episode,Gamma,Episode Length,Average time,Average reward
print("Gamma:", mc_parameters_optimal_values['Gamma'])
print("Episode:", mc_parameters_optimal_values['Episode'])
print("Episode Length:", mc_parameters_optimal_values['Episode Length'])
print("Average Reward:", mc_parameters_optimal_values['Average reward'])
print("Average Time:", mc_parameters_optimal_values['Average time'])
print('\n')
print('\n')

print("Optimal Parameters of Q-Learning:")
#Episode,Gamma,Alpha,Epsilon,Average time,Average reward
print("Gamma:", q_parameters_optimal_values['Gamma'])
print("Alpha:", q_parameters_optimal_values['Alpha'])
print("Epsilon:", q_parameters_optimal_values['Epsilon'])
print("Episode:", q_parameters_optimal_values['Episode'])
print("Average Reward:", q_parameters_optimal_values['Average reward'])
print("Average Time:", q_parameters_optimal_values['Average time'])
print('\n')
print('\n')

print("Optimal Parameters of Double-Q-Learning:")
#Episode,Gamma,Alpha,Epsilon,Average time,Average reward
print("Gamma:", dq_parameters_optimal_values['Gamma'])
print("Alpha:", dq_parameters_optimal_values['Alpha'])
print("Epsilon:", dq_parameters_optimal_values['Epsilon'])
print("Episode:", dq_parameters_optimal_values['Episode'])
print("Average Reward:", dq_parameters_optimal_values['Average reward'])
print("Average Time:", dq_parameters_optimal_values['Average time'])
print('\n')
print('\n')


print("Optimal Parameters of Triple-Q-Learning:")
#Episode,Gamma,Alpha,Epsilon,Average time,Average reward
print("Gamma:", tq_parameters_optimal_values['Gamma'])
print("Alpha:", tq_parameters_optimal_values['Alpha'])
print("Epsilon:", tq_parameters_optimal_values['Epsilon'])
print("Episode:", tq_parameters_optimal_values['Episode'])
print("Average Reward:", tq_parameters_optimal_values['Average reward'])
print("Average Time:", tq_parameters_optimal_values['Average time'])


# Create a new DataFrame to hold the optimal values
optimal_values_df = pd.DataFrame({
    'Method': ['Policy Iteration', 'Value Iteration','Monte Carlo ES', 'Q-Learning', 'Double Q-Learning', 'Triple Q-Learning',],
    'Optimal Parameters': [
        p_parameters_optimal_values.to_dict(),
        v_parameters_optimal_values.to_dict(),
        mc_parameters_optimal_values.to_dict(),
        q_parameters_optimal_values.to_dict(),
        dq_parameters_optimal_values.to_dict(),
        tq_parameters_optimal_values.to_dict()

    ]
})

# Saving the DataFrame as a CSV file
optimal_values_df.to_csv('data/optimal_parameters_summary_NEW.csv', index=False)