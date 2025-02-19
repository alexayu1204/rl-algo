import copy
import pickle
from collections import defaultdict

import gym
import numpy as np
import time
from tqdm import tqdm
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

from rl2023.constants import EX5_BIPEDAL_CONSTANTS as BIPEDAL_CONSTANTS
from rl2023.exercise4.agents import DDPG
from rl2023.exercise4.train_ddpg import train
from rl2023.exercise3.replay import ReplayBuffer
from rl2023.util.hparam_sweeping import generate_hparam_configs
from rl2023.util.result_processing import Run

from skopt import Optimizer
from skopt.utils import create_result

RENDER = False
SWEEP = False # TRUE TO SWEEP OVER POSSIBLE HYPERPARAMETER CONFIGURATIONS
NUM_SEEDS_SWEEP = 10 # NUMBER OF SEEDS TO USE FOR EACH HYPERPARAMETER CONFIGURATION
SWEEP_SAVE_RESULTS = True # TRUE TO SAVE SWEEP RESULTS TO A FILE
SWEEP_SAVE_ALL_WEIGTHS = False # TRUE TO SAVE ALL WEIGHTS FROM EACH SEED
ENV = "BIPEDAL" # "ACROBOT" is also possible if you uncomment the corresponding code, but is not assessed for DQN.

# IN EXERCISE 5 YOU SHOULD TUNE PARAMETERS IN THIS CONFIG ONLY
BIPEDAL_CONFIG = {
    "policy_learning_rate": 1e-4,
    "critic_learning_rate": 1e-3,
    "critic_hidden_size": [64, 64],
    "policy_hidden_size": [64, 64],
    "gamma": 0.99,
    "tau": 0.2,
    "batch_size": 64,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
}
BIPEDAL_CONFIG.update(BIPEDAL_CONSTANTS)

### INCLUDE YOUR CHOICE OF HYPERPARAMETERS HERE ###
BIPEDAL_HPARAMS = {...}

SWEEP_RESULTS_FILE_BIPEDAL = "DDPG-Bipedal-sweep-results-ex5.pkl"

# Define the search space for the hyperparameters
search_space = [
    ("policy_learning_rate", 1e-5, 1e-2),
    ("batch_size", 32, 512),
    ("tau", 1e-3, 1.0),
]

# Objective function for Bayesian optimization
def objective_function(params):
    learning_rate, batch_size, tau = params

    config = copy.deepcopy(BIPEDAL_CONFIG)
    config["policy_learning_rate"] = learning_rate
    config["batch_size"] = int(batch_size)
    config["tau"] = tau

    run = Run(config)
    for i in range(NUM_SEEDS_SWEEP):
        eval_returns, eval_timesteps, times, run_data = train(env, run.config, output=False)
        run.update(eval_returns, eval_timesteps, times, run_data)

    return -run.final_return_mean

# Main code
if __name__ == "__main__":
    if ENV == "BIPEDAL":
        CONFIG = BIPEDAL_CONFIG
        HPARAMS_SWEEP = BIPEDAL_HPARAMS
        SWEEP_RESULTS_FILE = SWEEP_RESULTS_FILE_BIPEDAL
    else:
        raise (ValueError(f"Unknown environment {ENV}"))

    env = gym.make(CONFIG["env"])

    if SWEEP:
        # Initialize the Bayesian optimizer
        optimizer = Optimizer(
            dimensions=[(min_val, max_val) for _, min_val, max_val in search_space],
            base_estimator="gp",
            n_initial_points=10,
        )

        # Perform Bayesian optimization
        for i in range(50):
            suggested_params = optimizer.ask()
            suggested_params_dict = {name: value for (name, _, _), value in zip(search_space, suggested_params)}

            print(f"\nStarting new run with hyperparameters: {suggested_params_dict}")
            fitness = objective_function(suggested_params)
            optimizer.tell(suggested_params, fitness)
            print(f"Finished run. Mean final score: {-fitness}")

        # Save the results
        if SWEEP_SAVE_RESULTS:
            print(f"Saving results to {SWEEP_RESULTS_FILE}")
            result = create_result(optimizer.Xi, optimizer.yi, optimizer.space, optimizer.models)
            with open(SWEEP_RESULTS_FILE, "wb") as f:
                pickle.dump(result, f)
    else:
        _ = train(env, CONFIG)

    env.close()
