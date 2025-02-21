#!/usr/bin/env python
"""
Grid-sweep hyperparameter tuning for DDPG on BipedalWalker-v3 (Exercise 5).

This script builds on the baseline train(env, config) function from exercise4.train_ddpg.py.
It uses EX5_BIPEDAL_CONSTANTS (from constants.py) and additional hyperparameter ranges
to perform a grid sweep via generate_hparam_configs and Run objects.
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
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

RENDER = False
SWEEP = True         # Set to True to perform grid sweep.
NUM_SEEDS_SWEEP = 1  # Number of seeds per hyperparameter configuration.
SWEEP_SAVE_RESULTS = True  # Save sweep results.
SWEEP_SAVE_ALL_WEIGTHS = True  # Save all weights for each seed.
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

BIPEDAL_HPARAMS = {
    "critic_hidden_size": [[64, 64]],
    "policy_hidden_size": [[64, 64]],
    "batch_size": [256],
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
                eval_returns, eval_timesteps, times, run_data = train(env, run.config)
                run.update(eval_returns, eval_timesteps, times, run_data)
            results.append(copy.deepcopy(run))
            print(f"Finished run with hyperparameters {hparams_values}. "
                  f"Mean final score: {run.final_return_mean} +- {run.final_return_ste}")
        if SWEEP_SAVE_RESULTS:
            print(f"Saving results to {SWEEP_RESULTS_FILE}")
            with open(SWEEP_RESULTS_FILE, 'wb') as f:
                pickle.dump(results, f)
    else:
        _ = train(env, CONFIG)
    env.close()
