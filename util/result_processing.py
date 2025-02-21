import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

class Run:
    def __init__(self, config: Dict):
        self._config = config
        self._run_name = None
        self._final_returns = []
        self._train_times = []
        self._run_data = []
        self._agent_weights_filenames = []
        self._run_ids = []
        self._all_eval_timesteps = []
        self._all_returns = []

    def update(self, eval_returns, eval_timesteps, times=None, run_data=None):
        self._run_ids.append(len(self._run_ids))
        if self._config.get('save_filename') is not None:
            self._agent_weights_filenames.append(self._config['save_filename'])
            self._config['save_filename'] = None

        self._all_eval_timesteps.append(eval_timesteps)
        self._all_returns.append(eval_returns)
        if isinstance(eval_returns, dict):
            returns_list = eval_returns.get("train_ep_returns", [])
        else:
            returns_list = eval_returns

        try:
            self._final_returns.append(returns_list[-1])
        except (IndexError, TypeError):
            self._final_returns.append(0)
        if times is not None:
            try:
                self._train_times.append(times[-1])
            except (IndexError, TypeError):
                pass
        if run_data is not None:
            self._run_data.append(run_data)

    def set_save_filename(self, filename):
        if self._config.get("save_filename") is not None:
            print(f"Warning: Save filename already set in config. Overwriting to {filename}.")
        self._config['save_filename'] = f"{filename}.pt"

    @property
    def run_name(self):
        return self._run_name

    @run_name.setter
    def run_name(self, name):
        self._run_name = name

    @property
    def final_return_mean(self) -> float:
        final_returns = np.array(self._final_returns)
        return final_returns.mean()

    @property
    def final_return_ste(self) -> float:
        final_returns = np.array(self._final_returns)
        return np.std(final_returns, ddof=1) / np.sqrt(np.size(final_returns))

    @property
    def final_returns(self) -> np.ndarray:
        return np.array(self._final_returns)

    @property
    def train_times(self) -> np.ndarray:
        return np.array(self._train_times)

    @property
    def config(self):
        return self._config

    @property
    def run_ids(self) -> List[int]:
        return self._run_ids

    @property
    def agent_weights_filenames(self) -> List[str]:
        return self._agent_weights_filenames

    @property
    def run_data(self) -> List[Dict]:
        return self._run_data

    @property
    def all_eval_timesteps(self) -> np.ndarray:
        return np.array(self._all_eval_timesteps)

    @property
    def all_returns(self) -> np.ndarray:
        return np.array(self._all_returns)

def rank_runs(runs: List[Run]):
    return sorted(runs, key=lambda x: x.final_return_mean, reverse=True)

def get_best_saved_run(runs: List[Run]) -> Tuple[Run, str]:
    ranked_runs = rank_runs(runs)
    best_run = ranked_runs[0]
    if best_run.agent_weights_filenames:
        best_run_id = np.argmax(best_run.final_returns)
        return best_run, best_run.agent_weights_filenames[best_run_id]
    else:
        raise ValueError(f"No saved runs found for highest mean final returns run {best_run.run_name}.")
