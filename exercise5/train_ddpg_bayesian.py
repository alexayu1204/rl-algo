#!/usr/bin/env python
"""
Bayesian optimization for tuning hyperparameters of DDPG on BipedalWalker-v3 (Exercise 5).

This script uses scikit-optimize to search over a continuous hyperparameter space by
calling the baseline train(env, config) function from exercise4/train_ddpg.py.
Results are saved to a pickle file.
"""

import copy
import pickle
import random
from collections import defaultdict

import gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from constants import EX5_BIPEDAL_CONSTANTS as BIPEDAL_CONSTANTS
from exercise4.train_ddpg import train  # Baseline train(env, config)
from util.result_processing import Run

from skopt import Optimizer
from skopt.space import Real, Integer

# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)

RENDER = False
SWEEP = True  # Use Bayesian optimization.
NUM_SEEDS_SWEEP = 3  # Number of seeds per configuration for robustness
SWEEP_SAVE_RESULTS = True
SWEEP_SAVE_ALL_WEIGTHS = False
ENV = "BIPEDAL"

# Base configuration for Exercise 5:
BIPEDAL_CONFIG = {
    "env": "BipedalWalker-v3",
    "eval_freq": 10000,
    "eval_episodes": 5,
    "target_return": 300.0,
    "episode_length": 1600,
    "max_timesteps": 250000,
    "max_time": 60 * 60,
    "policy_learning_rate": 1e-4,         # initial default value (will be tuned)
    "critic_learning_rate": 1e-4,           # fixed for now
    "gamma": 0.99,
    "tau": 0.005,                         # fixed for now
    "batch_size": 256,                    # default value (to be tuned)
    "buffer_capacity": int(1e6),
    "save_filename": "bipedal_q5_latest.pt",
    "algo": "DDPG",
    "warmup_steps": 500,
    "policy_hidden_size": [400, 300],
    "critic_hidden_size": [400, 300],
}
# BIPEDAL_CONFIG.update(BIPEDAL_CONSTANTS)

# Define the search space using skopt.space objects:
search_space = [
    Real(1e-5, 1e-3, prior="log-uniform", name="policy_learning_rate"),
    Integer(128, 256, name="batch_size"),
    Real(0.001, 0.01, name="tau"),
]

def objective_function(params):
    # The parameters come in the order defined above:
    policy_lr, batch_size, tau = params
    config = copy.deepcopy(BIPEDAL_CONFIG)
    config["policy_learning_rate"] = policy_lr
    config["batch_size"] = int(batch_size)
    config["tau"] = tau

    run = Run(config)
    # Run multiple seeds per configuration
    for i in range(NUM_SEEDS_SWEEP):
        # Optionally, you could set the seed inside train() as well.
        eval_returns, eval_timesteps, times, run_data = train(env, run.config)
        run.update(eval_returns, eval_timesteps, times, run_data)
    # We want to maximize final return so we return negative mean for minimization.
    return -run.final_return_mean

if __name__ == "__main__":
    if ENV == "BIPEDAL":
        CONFIG = BIPEDAL_CONFIG
    else:
        raise ValueError(f"Unknown environment {ENV}")

    env = gym.make(CONFIG["env"])
    
    if SWEEP:
        # Create the Bayesian optimizer with the defined search space:
        optimizer = Optimizer(
            dimensions=search_space,
            base_estimator="gp",
            n_initial_points=10,
            random_state=seed
        )
        n_iterations = 30
        for i in range(n_iterations):
            suggested_params = optimizer.ask()
            print(f"\nIteration {i+1}/{n_iterations}, suggested params: {suggested_params}")
            fitness = objective_function(suggested_params)
            optimizer.tell(suggested_params, fitness)
            print(f"Iteration {i+1} completed. Mean final score: {-fitness:.2f}")
        # Save the optimization results.
        SWEEP_RESULTS_FILE = "DDPG-Bipedal-sweep-results-ex5.pkl"
        print(f"\nSaving Bayesian optimization results to {SWEEP_RESULTS_FILE}")
        result = create_result(optimizer.Xi, optimizer.yi, optimizer.space, optimizer.models)
        with open(SWEEP_RESULTS_FILE, "wb") as f:
            pickle.dump(result, f)
    else:
        _ = train(env, CONFIG)
    env.close()
