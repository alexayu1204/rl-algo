#!/usr/bin/env python
"""
Grid-sweep hyperparameter tuning for DDPG on BipedalWalker-v3 (Exercise 5).

This script calls the baseline train(env, config) function from exercise4/train_ddpg.py
and sweeps a smaller set of hyperparameters for quicker experiments.
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
from util.hparam_sweeping import generate_hparam_configs
from util.result_processing import Run

RENDER = False
SWEEP = True         # Perform grid sweep.
NUM_SEEDS_SWEEP = 1  # Number of seeds per configuration.
SWEEP_SAVE_RESULTS = True
SWEEP_SAVE_ALL_WEIGTHS = True
ENV = "BIPEDAL"

# Base configuration for a shorter run and simpler search:
BIPEDAL_CONFIG = {
    "env": "BipedalWalker-v3",
    "eval_freq": 10000,
    "eval_episodes": 5,
    "target_return": 300.0,
    "episode_length": 1600,
    "max_timesteps": 400000,    # Reduced from 250k to 100k for faster experiments
    "max_time": 60 * 60,        # 1 hour max time
    "policy_learning_rate": 1e-4,
    "critic_learning_rate": 1e-3,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 128,
    "buffer_capacity": int(1e6),
    "save_filename": "bipedal_q5_latest.pt",
    "algo": "DDPG",
    "warmup_steps": 500,       # Fewer warmup steps
    "policy_hidden_size": [256, 256],
    "critic_hidden_size": [256, 256],
    "actor_lr_start": 1e-4,
    "actor_lr_end":   1e-5,
    "critic_lr_start":1e-3,
    "critic_lr_end":  1e-4,
    "noise_std_start":0.2,
    "noise_std_end":  0.05,
}
# BIPEDAL_CONFIG.update(BIPEDAL_CONSTANTS)

# Best for 400K epochs returns max >280:
# "actor_lr_start": [1e-3],
# "critic_lr_start": [1e-4],
# "max_timesteps": [400000],

# NARROWER hyperparameter grid for quicker experiments
BIPEDAL_HPARAMS = {
    "max_timesteps": [400000], 
    "actor_lr_start": [1e-3, 1e-4],
    "critic_lr_start": [1e-4],
    # "noise_std_start":[0.2],
    # "noise_std_end":  [0.05],
    "policy_hidden_size": [[400, 300]],
    "critic_hidden_size": [[400, 300]],
    # "batch_size": [128, 256],
    # "policy_learning_rate": [1e-4, 5e-5],
    # "critic_learning_rate": [1e-3],   # Keep the critic LR fixed at 1e-3
    # "tau": [0.005],                  # Keep tau fixed to 0.005 for now
}

SWEEP_RESULTS_FILE_BIPEDAL = "DDPG-Bipedal-sweep-results-ex5.pkl"

if __name__ == "__main__":
    if ENV == "BIPEDAL":
        CONFIG = BIPEDAL_CONFIG
        HPARAMS_SWEEP = BIPEDAL_HPARAMS
        SWEEP_RESULTS_FILE = SWEEP_RESULTS_FILE_BIPEDAL
    else:
        raise ValueError(f"Unknown environment {ENV}")

    env = gym.make(CONFIG["env"])
    if SWEEP and HPARAMS_SWEEP is not None:
        # Generate configurations from the narrower hyperparameter search space
        config_list, swept_params = generate_hparam_configs(CONFIG, HPARAMS_SWEEP)
        results = []
        for config in config_list:
            run = Run(config)
            hparams_values = '_'.join([':'.join([key, str(config[key])]) for key in swept_params])
            run.run_name = hparams_values
            print(f"\nStarting new run with hyperparameters: {hparams_values}")
            for i in range(NUM_SEEDS_SWEEP):
                print(f"Training iteration: {i + 1}/{NUM_SEEDS_SWEEP}")
                run_save_filename = '--'.join([run.config["algo"], run.config["env"], hparams_values, str(i)])
                if SWEEP_SAVE_ALL_WEIGTHS:
                    run.set_save_filename(run_save_filename)
                # Call baseline train(env, config)
                eval_returns, eval_timesteps, times, run_data = train(env, run.config)
                run.update(eval_returns, eval_timesteps, times, run_data)
            results.append(copy.deepcopy(run))
            print(f"Finished run with hyperparameters {hparams_values}. "
                  f"Mean final score: {run.final_return_mean:.2f} ± {run.final_return_ste:.2f}")
        if SWEEP_SAVE_RESULTS:
            print(f"Saving results to {SWEEP_RESULTS_FILE}")
            with open(SWEEP_RESULTS_FILE, 'wb') as f:
                pickle.dump(results, f)
    else:
        # Just run a single training session if SWEEP=False
        _ = train(env, CONFIG)
    env.close()
