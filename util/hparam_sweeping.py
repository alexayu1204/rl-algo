import itertools
import torch
from typing import Dict, List, Tuple, Iterable
import numpy as np


def generate_hparam_configs(base_config:Dict, hparam_ranges:Dict) -> Tuple[List[Dict], List[str]]:
    """
    Generate a list of hyperparameter configurations for hparam sweeping

    :param base_config (Dict): base configuration dictionary
    :param hparam_ranges (Dict): dictionary mapping hyperparameter names to lists of values to sweep over
    :return (Tuple[List[Dict], List[str]]): list of hyperparameter configurations and swept parameter names
    """

    keys, values = zip(*hparam_ranges.items())
    hparam_configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    swept_params = list(hparam_ranges.keys())

    new_configs = []
    for hparam_config in hparam_configurations:
        new_config = base_config.copy()
        new_config.update(hparam_config)
        new_configs.append(new_config)

    return new_configs, swept_params


def grid_search(num_samples: int, min: float = None, max: float = None, **kwargs)->Iterable:
    """ Implement this method to set hparam range over a grid of hyperparameters.
    :param num_samples (int): number of samples making up the grid
    :param min (float): minimum value for the allowed range to sweep over
    :param max (float): maximum value for the allowed range to sweep over
    :param kwargs: additional keyword arguments to parametrise the grid.
    :return (Iterable): tensor/array/list/etc... of values to sweep over

    Example use: hparam_ranges['batch_size'] = grid_search(64, 512, 6, log=True)

    **YOU MAY IMPLEMENT THIS FUNCTION FOR Q5**

    """
    log = False
    if log:
        values = np.logspace(np.log10(min), np.log10(max), num_samples)
    else:
        values = np.linspace(min, max, num_samples)
    return torch.tensor(values)


def random_search(num_samples: int, distribution: str, min: float=None, max: float=None, **kwargs) -> Iterable:
    """ Implement this method to sweep via random search, sampling from a given distribution.
    :param num_samples (int): number of samples to take from the distribution
    :param distribution (str): name of the distribution to sample from
        (you can instantiate the distribution using torch.distributions, numpy.random, or else).
    :param min (float): minimum value for the allowed range to sweep over (for continuous distributions)
    :param max (float): maximum value for the allowed range to sweep over (for continuous distributions)
    :param kwargs: additional keyword arguments to parametrise the distribution.

    Example use: hparam_ranges['lr'] = random_search(1e-6, 1e-1, 10, distribution='exponential', lambda=0.1)

    **YOU MAY IMPLEMENT THIS FUNCTION FOR Q5**

    """
    if distribution == 'uniform':
        values = np.random.uniform(min, max, num_samples)
    elif distribution == 'exponential':
        values = np.random.exponential(scale=kwargs['lambda'], size=num_samples)
    elif distribution == 'loguniform':
        values = np.exp(np.random.uniform(np.log(min), np.log(max), num_samples))
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    return torch.tensor(values)


# from skopt import BayesSearchCV
# from skopt.space import Real, Integer, Categorical

# def bayesian_optimization(estimator, search_spaces, n_iter=50, cv=5, n_jobs=-1, verbose=0):
#     """Now, let's implement Bayesian optimization using scikit-optimize library: !pip install scikit-optimize
#     For DDPG, Bayesian optimization is a suitable method for hyperparameter tuning because it can efficiently
#     search the hyperparameter space by building a probabilistic model of the objective function and using it 
#     to select the most promising hyperparameters to evaluate in the true objective function. 
#     This can be beneficial for DDPG because it has many hyperparameters, and the objective function 
#     (i.e., the performance of the trained agent) can be expensive to evaluate."""
#     opt = BayesSearchCV(
#         estimator,
#         search_spaces,
#         n_iter=n_iter,
#         cv=cv,
#         n_jobs=n_jobs,
#         verbose=verbose
#     )
#     return opt
