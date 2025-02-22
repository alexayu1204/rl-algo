# Reinforcement Learning Implementations

This repository contains implementations of various reinforcement learning algorithms, with a focus on the Acrobot-v1 environment. The implementations include DQN and REINFORCE algorithms with several enhancements and optimizations.

## DDPG Hyperparameter Tuning for BipedalWalker-v3

This repository contains implementations for training and hyperparameter tuning of a Deep Deterministic Policy Gradient (DDPG) agent on the BipedalWalker-v3 environment. The project is divided into two main parts:

- A baseline implementation of DDPG that includes random warmup, soft target updates, and Gaussian exploration. This script accepts a configuration dictionary to control hyperparameters.
- An extension that implements hyperparameter tuning using both grid search and Bayesian optimization. In addition, evaluation scripts and analysis utilities are provided to help identify the best-performing checkpoints.

---

## TD3 (Twin Delayed DDPG)

TD3 is an enhanced version of DDPG designed to address its overestimation and instability issues. It incorporates three key improvements:

- **Clipped Double-Q Learning:**  
  Two critic networks are trained; the target Q-value is computed as the minimum of the two critics’ estimates, reducing overestimation bias.

- **Delayed Policy Updates:**  
  The actor (policy) is updated less frequently than the critics (e.g., one actor update for every two critic updates), which stabilizes training.

- **Target Policy Smoothing:**  
  Noise is added to the target action and then clipped. This smooths the Q-function, preventing the policy from exploiting errors in the value estimates.

Additional enhancements include:
- **Reward Normalization:** Using running mean and standard deviation to scale rewards.
- **Learning-Rate Scheduling:** Linearly decaying the actor and critic learning rates during training.
- **Layer Normalization & Orthogonal Initialization:** Applied to network layers for improved stability.

These techniques collectively result in a more robust and stable TD3 agent for continuous control tasks like BipedalWalker-v3.

---

## Table of Contents

- [Overview](#overview)
- [Folder Structure](#folder-structure)
- [Exercise 4: Baseline DDPG](#exercise-4-baseline-ddpg)
  - [Key Features](#key-features)
  - [Usage](#usage)
- [Exercise 5: Hyperparameter Tuning](#exercise-5-hyperparameter-tuning)
  - [Grid Search](#grid-search)
  - [Bayesian Optimization](#bayesian-optimization)
  - [Evaluation and Analysis](#evaluation-and-analysis)
- [Enhancements and Improvements](#enhancements-and-improvements)
- [Results and Experiments](#results-and-experiments)
- [How to Run](#how-to-run)
- [Future Directions](#future-directions)
- [License](#license)

---

## Overview

This project aims to improve the performance of a baseline DDPG agent on the BipedalWalker-v3 environment by systematically exploring hyperparameter spaces. The baseline (Exercise 4) is then extended in Exercise 5 with grid search and Bayesian optimization methods to automatically search for promising hyperparameter configurations. Key techniques include:

- **Learning-Rate Scheduling:** Linearly decaying the learning rates for both actor and critic.
- **Noise Decay:** Linearly reducing exploration noise over training to improve policy refinement.
- **Architectural Variations:** Tuning hidden layer sizes (e.g., [400,300] vs. [256,256]) and batch sizes.
- **Reduced Training Epochs:** Using a smaller number of total timesteps (e.g., 100k–250k) for quicker experiments.

These enhancements are inspired by state-of-the-art methods and practical experience in tuning DDPG for continuous control tasks.

---

## Folder Structure

```
.
├── constants.py                   # Contains all experiment constants and default hyperparameters.
├── exercise4/
│   ├── train_ddpg.py              # Baseline DDPG training script (accepts a configuration dictionary).
│   └── evaluate_ddpg.py           # Script to evaluate a checkpoint.
├── exercise5/
│   ├── train_ddpg.py              # Grid search hyperparameter tuning script.
│   ├── train_ddpg_bayesian.py     # Bayesian optimization script for hyperparameter tuning.
│   └── evaluate_ddpg.py           # Evaluation script (can also be used from exercise4).
└── util/
    ├── hparam_sweeping.py         # Utility for generating hyperparameter configurations.
    └── result_processing.py       # Utility for processing and ranking experiment results.
```

---

## Exercise 4: Baseline DDPG

### Key Features

- **Configurable Hyperparameters:** Accepts a configuration dictionary with keys such as learning rates, hidden sizes, batch size, warmup steps, etc.
- **Random Warmup:** Uses random actions for a fixed number of steps to populate the replay buffer.
- **Soft Target Updates:** Implements soft updates for both actor and critic target networks.
- **Gaussian Exploration:** Uses a diagonal Gaussian noise process for exploration.
- **Learning-Rate Scheduling and Noise Decay:** (Enhanced version) Linearly decays learning rates and exploration noise over training.

### Usage

Run the baseline training with:
```bash
PYTHONPATH=. python exercise4/train_ddpg.py
```
Or, call the `train(env, config)` function from your own script.

---

## Exercise 5: Hyperparameter Tuning

Exercise 5 extends the baseline with automated hyperparameter tuning.

### Grid Search

- **File:** `exercise5/train_ddpg.py`
- **Purpose:** Run a grid search over a narrowed set of hyperparameters for faster experimentation.
- **Search Space:**  
  - **Policy Hidden Size:** `[[256,256], [400,300]]`
  - **Critic Hidden Size:** `[[256,256], [400,300]]`
  - **Batch Size:** `[128, 256]`
  - **Policy Learning Rate:** `[1e-4, 5e-5]`
  - **Critic Learning Rate:** Fixed at `1e-3`
  - **Tau:** Fixed at `0.005`
- **Total Training Timesteps:** 100,000 (for faster convergence during experiments)
- **Warmup Steps:** 500
- **Output:** Checkpoint files for each configuration and a results pickle file (`DDPG-Bipedal-sweep-results-ex5.pkl`)

### Bayesian Optimization

- **File:** `exercise5/train_ddpg_bayesian.py`
- **Purpose:** Use Bayesian optimization (via scikit-optimize) to search over a continuous hyperparameter space.
- **Search Space (using skopt):**
  - **Policy Learning Rate:** Real, log-uniform between `1e-5` and `1e-3`
  - **Batch Size:** Integer between `128` and `256`
  - **Tau:** Real between `0.001` and `0.01`
- **Evaluation:** Each configuration is evaluated over multiple seeds, and the negative mean final return is used as the objective.
- **Output:** Saves results to a pickle file for later analysis.

### Evaluation and Analysis

- **Evaluation Script:** `exercise5/evaluate_ddpg.py` provides a command-line interface to evaluate a saved checkpoint.
- **Analysis Script (Optional):** You can create an analysis script (e.g., `analyze_results.py`) to load the sweep results pickle file and rank runs to identify the best-performing configuration.

---

## Enhancements and Improvements

- **Learning-Rate Scheduling:**  
  The baseline now decays both the actor and critic learning rates linearly from a starting value to an ending value over the course of training.

- **Noise Decay:**  
  The exploration noise standard deviation is also decayed linearly from an initial high value to a lower value as training progresses.

- **Narrower Hyperparameter Grid:**  
  The grid search has been refined to explore only the most promising hyperparameter ranges (e.g., hidden sizes [256,256] vs. [400,300], learning rates, batch sizes) to reduce training time and get meaningful differences quickly.

- **Modularity:**  
  The code is modular, with baseline training in Exercise 4 and hyperparameter tuning in Exercise 5. Results and logs are saved for later analysis.

---

## Results and Experiments

After training with these enhanced methods, you should obtain multiple checkpoint files with different hyperparameter configurations. The results file (`DDPG-Bipedal-sweep-results-ex5.pkl`) contains Run objects that record the final return, training time, and configuration details. Use the analysis script to rank these runs and then evaluate the best checkpoint using the provided evaluation script.

**Note:**  
If the returns remain low (e.g., under -100), consider further adjustments such as increasing total timesteps, fine-tuning the learning-rate schedules, or even trying advanced methods (TD3, SAC) that are known to perform better on BipedalWalker-v3.

---

## How to Run

### Training (Grid Search)
```bash
PYTHONPATH=. python exercise5/train_ddpg.py
```
This will run the grid search over the specified hyperparameter combinations and save the results.

### Training (Bayesian Optimization)
```bash
PYTHONPATH=. python exercise5/train_ddpg_bayesian.py
```
This will run Bayesian optimization over the defined search space and save the results.

### Evaluation
```bash
PYTHONPATH=. python exercise5/evaluate_ddpg.py --checkpoint_path <checkpoint_file> --episodes 10 --render --env_mode human
```
Replace `<checkpoint_file>` with the path to your best checkpoint file as determined by the analysis.

### Analysis (Optional)
Create and run an analysis script (e.g., `analyze_results.py`) to load and rank your sweep results:
```bash
PYTHONPATH=. python analyze_results.py
```

---

## Future Directions

- **Further Tuning:**  
  If performance is still unsatisfactory, consider:
  - Increasing total training timesteps.
  - Experimenting with different network architectures.
  - Incorporating reward normalization.
  - Trying alternative algorithms (e.g., TD3 or SAC).

- **Advanced Bayesian Methods:**  
  You could also integrate more advanced Bayesian optimization frameworks or multi-fidelity methods to reduce the computational cost further.

---

## Requirements

- Python 3.7+
- PyTorch
- Gym (with BipedalWalker-v3 support)
- scikit-optimize
- tqdm
- numpy
- matplotlib

---

## Setup Instructions

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install required packages:
   ```bash
   pip install torch gym scikit-optimize tqdm numpy matplotlib
   ```
4. Run training and evaluation scripts as described above.


## Other implemented RL algorithms

## Table of Contents
- [Installation](#installation)
- [Algorithms](#algorithms)
  - [REINFORCE](#reinforce)
  - [DQN](#dqn)
- [Environments](#environments)
- [Performance Improvements](#performance-improvements)
- [Usage](#usage)
- [Results](#results)

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repo-name]
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Algorithms

### REINFORCE
The REINFORCE implementation includes several enhancements:

1. **Network Architecture**:
   - Larger network capacity [256, 128]
   - Layer Normalization for better training stability
   - GELU activation functions
   - Dropout layers (0.1) for regularization
   - Orthogonal initialization with environment-specific gains

2. **Training Dynamics**:
   - Increased learning rate (3e-3)
   - Cosine learning rate decay with warmup
   - Adaptive entropy coefficient
   - Gradient clipping
   - GAE (Generalized Advantage Estimation)

3. **Reward Shaping**:
   - Progressive height-based rewards
   - Momentum-based rewards
   - Action diversity encouragement
   - Consistency bonuses
   - Smart penalties for stillness

### DQN
The DQN implementation features:

1. **Network Architecture**:
   - Double DQN with target network
   - Efficient architecture with Layer Normalization
   - ReLU activation functions
   - Orthogonal initialization

2. **Training Improvements**:
   - Dynamic reward normalization
   - Soft target updates
   - Huber loss for stability
   - Gradient clipping
   - Optimized replay buffer

## Environments

### Acrobot-v1
- **State Space**: 6-dimensional
- **Action Space**: 3 discrete actions
- **Goal**: Swing the end of the double pendulum above the base
- **Success Threshold**: -63 (average reward)

### Other Supported Environments
- CartPole-v1
- LunarLander-v2

## Performance Improvements

### REINFORCE Enhancements
1. **Reward Shaping**:
   ```python
   if height > best_height:
       height_improvement = height - best_height
       reward += 0.3 * height_improvement
       best_height = height
       consecutive_good_actions += 1
   ```

2. **Momentum Rewards**:
   ```python
   if height > -1.5:
       momentum_reward = 0.15 * (abs(angular_vel1) + abs(angular_vel2))
       if height > 0:
           momentum_reward *= 1.5
       reward += momentum_reward
   ```

3. **Action Diversity**:
   ```python
   if last_action == action:
       action_repeat_count += 1
       if action_repeat_count > 2:
           reward -= 0.2 * action_repeat_count
   ```

### Architecture Improvements
```python
self.policy = nn.Sequential(
    nn.Linear(STATE_SIZE, hidden_size[0]),
    nn.LayerNorm(hidden_size[0]),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(hidden_size[0], hidden_size[1]),
    nn.LayerNorm(hidden_size[1]),
    nn.GELU(),
    nn.Linear(hidden_size[1], ACTION_SIZE)
)
```

## Usage

### Training REINFORCE
```bash
# Train REINFORCE on Acrobot-v1
python exercise3/train_reinforce.py --env Acrobot-v1

# Train with specific seed
python exercise3/train_reinforce.py --env Acrobot-v1 --seed 42
```

### Training DQN
```bash
# Train DQN on Acrobot-v1
python exercise3/train_dqn.py --env Acrobot-v1

# Train with rendering disabled
python exercise3/train_dqn.py --env Acrobot-v1 --no-render
```

### Evaluation
```bash
# Evaluate REINFORCE
python exercise3/evaluate_reinforce.py --env Acrobot-v1 --episodes 10

# Evaluate DQN
python exercise3/evaluate_dqn.py --env Acrobot-v1 --episodes 10
```

## Results

### Acrobot-v1 Performance
- **Mean Return**: -111.50
- **Standard Deviation**: 30.08
- **Best Episode**: -82.00
- **Worst Episode**: -194.00
- **Success Rate**: Episodes with return > -120: 80%

### Training Parameters
```python
ENV_CONFIGS = {
    "Acrobot-v1": {
        "max_steps": 500,
        "success_threshold": -63,
        "learning_rate": 3e-3,
        "hidden_size": [256, 128],
        "gamma": 0.995,
        "max_timesteps": 200000,
        "eval_freq": 2000,
        "eval_episodes": 5,
        "max_time": 7200,
    }
}
```