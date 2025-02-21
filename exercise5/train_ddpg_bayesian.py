#!/usr/bin/env python
"""
Bayesian optimization for tuning hyperparameters of DDPG on BipedalWalker-v3 (Exercise 5).

This script uses scikit-optimize to search over a continuous hyperparameter space by
calling the baseline train(env, config) function from exercise4.train_ddpg.py.
Results are saved to a pickle file.
"""

import copy
import pickle
from collections import defaultdict

import gym
import numpy as np
from tqdm import tqdm
from typing import Dict
import matplotlib.pyplot as plt

from constants import EX5_BIPEDAL_CONSTANTS as BIPEDAL_CONSTANTS
from exercise4.train_ddpg import train  # Baseline train(env, config)
from util.result_processing import Run

from skopt import Optimizer
from skopt.utils import create_result
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

RENDER = False
SWEEP = False  # Use Bayesian optimization.
NUM_SEEDS_SWEEP = 10  # Number of seeds per hyperparameter configuration.
SWEEP_SAVE_RESULTS = True
SWEEP_SAVE_ALL_WEIGTHS = False
ENV = "BIPEDAL"

BIPEDAL_CONFIG = {
    "env": "BipedalWalker-v3",
    "eval_freq": 20000,
    "eval_episodes": 100,
    "target_return": 300.0,
    "episode_length": 1600,
    "max_timesteps": 1500000,
    "max_time": 360 * 60,
    "policy_learning_rate": 1e-4,
    "critic_learning_rate": 1e-3,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 64,
    "buffer_capacity": int(1e6),
    "save_filename": "bipedal_q5_latest.pt",
    "algo": "DDPG",
}
BIPEDAL_CONFIG.update(BIPEDAL_CONSTANTS)

SWEEP_RESULTS_FILE_BIPEDAL = "DDPG-Bipedal-sweep-results-ex5.pkl"

# Define search space: (name, min, max)
search_space = [
    ("policy_learning_rate", 1e-5, 1e-2),
    ("batch_size", 32, 512),
    ("tau", 1e-3, 1.0),
]

def objective_function(params):
    learning_rate, batch_size, tau = params
    config = copy.deepcopy(BIPEDAL_CONFIG)
    config["policy_learning_rate"] = learning_rate
    config["batch_size"] = int(batch_size)
    config["tau"] = tau
    run = Run(config)
    for i in range(NUM_SEEDS_SWEEP):
        eval_returns, eval_timesteps, times, run_data = train(env, run.config)
        run.update(eval_returns, eval_timesteps, times, run_data)
    return -run.final_return_mean

if __name__ == "__main__":
    if ENV == "BIPEDAL":
        CONFIG = BIPEDAL_CONFIG
    else:
        raise ValueError(f"Unknown environment {ENV}")

    env = gym.make(CONFIG["env"])
    if SWEEP:
        optimizer = Optimizer(
            dimensions=[(min_val, max_val) for _, min_val, max_val in search_space],
            base_estimator="gp",
            n_initial_points=10,
        )
        for i in range(50):
            suggested_params = optimizer.ask()
            suggested_params_dict = {name: value for (name, _, _), value in zip(search_space, suggested_params)}
            print(f"\nStarting new run with hyperparameters: {suggested_params_dict}")
            fitness = objective_function(suggested_params)
            optimizer.tell(suggested_params, fitness)
            print(f"Finished run. Mean final score: {-fitness:.2f}")
        if SWEEP_SAVE_RESULTS:
            print(f"Saving Bayesian optimization results to {SWEEP_RESULTS_FILE_BIPEDAL}")
            result = create_result(optimizer.Xi, optimizer.yi, optimizer.space, optimizer.models)
            with open(SWEEP_RESULTS_FILE_BIPEDAL, "wb") as f:
                pickle.dump(result, f)
    else:
        _ = train(env, CONFIG)
    env.close()
