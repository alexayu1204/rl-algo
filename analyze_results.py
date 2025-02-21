#!/usr/bin/env python
"""
This script loads the sweep results from DDPG-Bipedal-sweep-results-ex5.pkl,
ranks the runs by final return, and prints a summary.
It also shows which checkpoint file achieved the best result.
"""

import pickle
from util.result_processing import rank_runs, get_best_saved_run

def main():
    results_file = "DDPG-Bipedal-sweep-results-ex5.pkl"
    try:
        with open(results_file, "rb") as f:
            runs = pickle.load(f)
    except Exception as e:
        print("Error loading sweep results:", e)
        return

    # Rank runs (higher mean return is better)
    sorted_runs = rank_runs(runs)
    print("Ranked Runs (sorted by final return mean):")
    for run in sorted_runs:
        print(f"Run: {run.run_name} | Final Return Mean: {run.final_return_mean:.2f} ± {run.final_return_ste:.2f}")

    try:
        best_run, best_weights = get_best_saved_run(runs)
        print("\nBest Run:")
        print(f"  Run Name: {best_run.run_name}")
        print(f"  Final Return Mean: {best_run.final_return_mean:.2f} ± {best_run.final_return_ste:.2f}")
        print(f"  Best checkpoint file: {best_weights}")
    except Exception as e:
        print("Error determining best run:", e)

if __name__ == "__main__":
    main()

