import itertools
from typing import Dict, List, Tuple, Iterable
import numpy as np
import torch

def generate_hparam_configs(base_config: Dict, hparam_ranges: Dict) -> Tuple[List[Dict], List[str]]:
    keys, values = zip(*hparam_ranges.items())
    hparam_configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    swept_params = list(hparam_ranges.keys())
    new_configs = []
    for hparam_config in hparam_configurations:
        new_config = base_config.copy()
        new_config.update(hparam_config)
        new_configs.append(new_config)
    return new_configs, swept_params

def grid_search(num_samples: int, min: float, max: float, log: bool=False) -> Iterable:
    if log:
        values = np.logspace(np.log10(min), np.log10(max), num_samples)
    else:
        values = np.linspace(min, max, num_samples)
    return torch.tensor(values)
