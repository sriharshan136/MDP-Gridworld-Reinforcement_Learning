# MDP-Gridworld-Reinforcement_Learning

This repository contains a comprehensive implementation of Markov Decision Process (MDP) algorithms for solving Gridworld problems. It includes Policy Iteration, Value Iteration, Monte Carlo Learning, Q-Learning, and advanced variations like Double and Triple Q-Learning. The project also explores hyperparameter optimization and visualization techniques.

## Project Structure

- **`Gridworld/`**: Contains core `.py` files and results.
  - **`Data/`**: Input data for running `.py` files.
- **`Results_10K_Iterations/`**: Results (CSV files) after running algorithms for 10,000 iterations.
- **`Results_50K_Iterations/`**: Results (CSV files) after running algorithms for 50,000 iterations.
- **`Hyper_Parameters/`**: CSV files containing different hyperparameters.
- **`Final_Animation_Images/`**: Final animations and plots of Gridworld training with various algorithms.

## Instructions

1. Run the `Test-*.py` files to evaluate specific algorithms and generate results.
2. Execute `HyperParameters.py` to explore hyperparameters for each algorithm.
3. Use `GenerateFinalPlots.py` to create bar and convergence plots for comparing algorithm performance.

## Python File Descriptions

- **`GridWorld.py`**: Defines the `GridWorld` class for simulating a grid-world environment with rewards, transitions, random policies, and visualizations.
- **`GridWorldVisualizer.py`**: Creates animated visualizations of algorithm training on Gridworld.
- **`PolicyIteration.py`**: Implements Policy Iteration for optimizing policies through evaluation and improvement.
- **`Test-PolicyIteration.py`**: Applies Policy Iteration, visualizing results and saving performance metrics.
- **`ValueIteration.py`**: Implements Value Iteration for iteratively optimizing state values and policies.
- **`Test-ValueIteration.py`**: Applies Value Iteration, visualizing results and saving performance metrics.
- **`MCLearningES.py`**: Implements Monte Carlo Learning with Exploring Starts for MDPs.
- **`Test-MCLearningES.py`**: Evaluates Monte Carlo Learning, visualizing and saving metrics.
- **`QLearning.py`**: Implements Q-Learning, a model-free reinforcement learning algorithm.
- **`Test-QLearning.py`**: Evaluates Q-Learning, visualizing results and saving metrics.
- **`DoubleQLearning.py`**: Implements Double Q-Learning for enhanced stability.
- **`Test-DoubleQLearning.py`**: Evaluates Double Q-Learning, visualizing results and saving metrics.
- **`TripleQLearning.py`**: Implements Triple Q-Learning for further enhanced performance.
- **`Test-TripleQLearning.py`**: Evaluates Triple Q-Learning, visualizing results and saving metrics.
- **`HyperParameters.py`**: Explores and saves hyperparameters for Policy Iteration, Value Iteration, Monte Carlo, and Q-Learning.
- **`HyperParametersCompare.py`**: Loads hyperparameter results, selects optimal configurations, and summarizes them in a CSV file.

## Results

- Results from 10,000 and 50,000 iterations for all algorithms are saved as CSV files in their respective folders.
- Final animations and plots are available for comparing performance metrics, including rewards, time, and variance.

## Requirements

Install the required Python packages using:
```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/sriharshan136/MDP-Gridworld-Reinforcement_Learning.git
   cd MDP-Gridworld-RL
   ```
2. Run the desired test scripts or visualization tools:
   ```bash
   python Test-PolicyIteration.py
   python GenerateFinalPlots.py
   ```

## Contributions

Contributions are welcome! If you have suggestions for improvements or additional features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

