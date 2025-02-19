import copy
import pickle
import gym
import numpy as np
import time
import os
import argparse
from tqdm import tqdm
from typing import List, Tuple, Dict
from collections import defaultdict
import matplotlib.pyplot as plt
import torch

from exercise3.agents import DQN
from exercise3.replay import ReplayBuffer

# Environment-specific configurations (only discrete action spaces)
ENV_CONFIGS = {
    "CartPole-v1": {
        "max_steps": 500,
        "success_threshold": 295,
        "learning_rate": 1e-3,
        "hidden_size": [256, 256],
        "target_update_freq": 10,
        "batch_size": 256,
        "gamma": 0.99,
        "buffer_capacity": int(1e5),
        "epsilon_start": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "epsilon_decay_strategy": "exponential",
        "max_timesteps": 50000,
        "eval_freq": 1000,
        "eval_episodes": 10,
        "max_time": 3600,
        "tau": 0.01,
    },
    "Acrobot-v1": {
        "max_steps": 500,
        "success_threshold": -63,
        "learning_rate": 1e-3,
        "hidden_size": [64, 64],
        "target_update_freq": 5,
        "batch_size": 64,
        "gamma": 0.99,
        "buffer_capacity": int(1e5),
        "epsilon_start": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "epsilon_decay_strategy": "exponential",
        "max_timesteps": 100000,
        "eval_freq": 2000,
        "eval_episodes": 10,
        "max_time": 7200,
        "tau": 0.01,
    },
    "LunarLander-v2": {
        "max_steps": 1000,
        "success_threshold": 140,
        "learning_rate": 1e-3,
        "hidden_size": [128, 128],
        "target_update_freq": 5,
        "batch_size": 64,
        "gamma": 0.99,
        "buffer_capacity": int(1e5),
        "epsilon_start": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "epsilon_decay_strategy": "exponential",
        "max_timesteps": 200000,
        "eval_freq": 2000,
        "eval_episodes": 10,
        "max_time": 7200,
        "tau": 0.01,
    }
}

def play_episode(
    env,
    agent,
    replay_buffer,
    train=True,
    explore=True,
    render=False,
    max_steps=500,
    batch_size=64,
) -> Tuple[int, float, Dict]:
    """Play one episode and train the agent."""
    obs = env.reset()
    done = False
    episode_data = defaultdict(list)
    episode_return = 0
    episode_timesteps = 0

    while not done and episode_timesteps < max_steps:
        if render:
            env.render()

        # Select action
        action = agent.act(obs, explore=explore)
        
        # Execute action
        next_obs, reward, done, _ = env.step(action)
        
        episode_timesteps += 1
        episode_return += reward

        if train:
            # Store transition in replay buffer
            replay_buffer.add(
                obs=obs,
                action=action,
                next_obs=next_obs,
                reward=reward,
                done=done,
            )

            # Update agent if enough samples
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                update_info = agent.update(batch)
                for k, v in update_info.items():
                    episode_data[k].append(v)

        obs = next_obs

    episode_data["episode_return"] = episode_return
    episode_data["episode_timesteps"] = episode_timesteps

    return episode_timesteps, episode_return, episode_data


def evaluate(env: gym.Env, agent: DQN, num_episodes: int, max_steps: int) -> Tuple[float, float]:
    """Evaluate the agent for a given number of episodes."""
    returns = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_return = 0
        steps = 0

        while not done and steps < max_steps:
            action = agent.act(obs, explore=False)
            obs, reward, done, _ = env.step(action)
            episode_return += reward
            steps += 1

        returns.append(episode_return)

    return float(np.mean(returns)), float(np.std(returns))


def train(env: gym.Env, config: Dict, output: bool = True) -> Tuple[List[float], List[int], List[float], Dict]:
    """Training loop with environment-specific configurations and model saving."""
    timesteps_elapsed = 0
    max_steps = config["max_steps"]
    best_eval_return = -float('inf')
    best_model_path = f"best_{env.spec.id}_model.pt"
    latest_model_path = f"latest_{env.spec.id}_model.pt"
    patience = 5
    no_improvement_count = 0

    print("\nInitializing DQN agent...")
    agent = DQN(
        action_space=env.action_space,
        observation_space=env.observation_space,
        **config,
    )
    print("Initializing replay buffer...")
    replay_buffer = ReplayBuffer(config["buffer_capacity"], device=agent.device)

    eval_returns = []
    eval_timesteps = []
    times = []
    episode_returns = []
    start_time = time.time()

    print("\nStarting training loop...")
    with tqdm(total=config["max_timesteps"], disable=not output) as pbar:
        while timesteps_elapsed < config["max_timesteps"]:
            if time.time() - start_time > config["max_time"]:
                print(f"\nTraining ended after {time.time() - start_time:.2f}s due to time limit.")
                break

            agent.schedule_hyperparameters(timesteps_elapsed, config["max_timesteps"])
            
            # Play episode
            obs = env.reset()
            done = False
            episode_return = 0
            episode_timesteps = 0

            while not done and episode_timesteps < max_steps:
                action = agent.act(obs, explore=True)
                next_obs, reward, done, _ = env.step(action)
                
                replay_buffer.add(obs, action, next_obs, reward, done)
                
                if len(replay_buffer) >= config["batch_size"]:
                    batch = replay_buffer.sample(config["batch_size"])
                    agent.update(batch)

                obs = next_obs
                episode_return += reward
                episode_timesteps += 1
                timesteps_elapsed += 1
                pbar.update(1)

                if timesteps_elapsed >= config["max_timesteps"]:
                    break

            episode_returns.append(episode_return)

            # Show progress
            if len(episode_returns) % 25 == 0:
                recent_returns = episode_returns[-25:]
                mean_return = np.mean(recent_returns)
                std_return = np.std(recent_returns)
                print(f"\nEpisode {len(episode_returns)}")
                print(f"Last 25 episodes - Mean: {mean_return:.2f}, Std: {std_return:.2f}")
                print(f"Epsilon: {agent.epsilon:.3f}")

            # Evaluation phase
            if timesteps_elapsed % config["eval_freq"] < episode_timesteps:
                eval_return, eval_std = evaluate(env, agent, config["eval_episodes"], max_steps)
                eval_returns.append(eval_return)
                eval_timesteps.append(timesteps_elapsed)
                times.append(time.time() - start_time)
                
                print(f"\nEvaluation at timestep {timesteps_elapsed}:")
                print(f"Mean return: {eval_return:.2f} +/- {eval_std:.2f}")
                
                # Save best and latest models
                agent.save(latest_model_path)
                if eval_return > best_eval_return:
                    best_eval_return = eval_return
                    agent.save(best_model_path)
                    print(f"New best model saved with return: {best_eval_return:.2f}")
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # Early stopping check
                if eval_return >= config["success_threshold"]:
                    print(f"\nEnvironment solved with return {eval_return:.2f}!")
                    break
                elif no_improvement_count >= patience:
                    print(f"\nStopping early due to no improvement for {patience} evaluations")
                    break

    print("\nTraining Summary:")
    print(f"Environment: {env.spec.id}")
    print(f"Total episodes completed: {len(episode_returns)}")
    print(f"Best evaluation return: {best_eval_return:.2f}")
    print(f"Final epsilon value: {agent.epsilon:.4f}")
    print(f"Total time elapsed: {(time.time() - start_time):.2f}s")
    print(f"Best model saved to: {best_model_path}")
    print(f"Latest model saved to: {latest_model_path}")

    return eval_returns, eval_timesteps, times, {}


def get_args():
    parser = argparse.ArgumentParser(description='Train a DQN agent on various gym environments')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        choices=list(ENV_CONFIGS.keys()),
                        help='Gym environment to train on')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering during training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    # Get environment-specific configuration
    if args.env not in ENV_CONFIGS:
        raise ValueError(f"Environment {args.env} not supported. Choose from: {list(ENV_CONFIGS.keys())}")
    
    config = ENV_CONFIGS[args.env].copy()
    
    print(f"Training on environment: {args.env}")
    print("Configuration:", config)
    
    # Set up environment and seeds
    env = gym.make(args.env)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Run training
    eval_returns, eval_timesteps, times, _ = train(env, config)
    env.close()
