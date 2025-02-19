import copy
import pickle
import gym
import numpy as np
import time
from tqdm import tqdm
from typing import List, Tuple, Dict
from collections import defaultdict
import torch

from exercise3.agents import Reinforce

# Environment-specific configurations
ENV_CONFIGS = {
    "CartPole-v1": {
        "max_steps": 500,
        "success_threshold": 475,
        "learning_rate": 8e-4,
        "hidden_size": [64, 64],
        "gamma": 0.99,
        "max_timesteps": 50000,
        "eval_freq": 1000,
        "eval_episodes": 10,
        "max_time": 3600,
    },
    "Acrobot-v1": {
        "max_steps": 500,
        "success_threshold": -63,
        "learning_rate": 3e-3,  # Increased learning rate
        "hidden_size": [256, 128],  # Larger network
        "gamma": 0.995,  # Increased gamma for better long-term planning
        "max_timesteps": 200000,  # More training time
        "eval_freq": 2000,
        "eval_episodes": 5,
        "max_time": 7200,
    },
    "LunarLander-v2": {
        "max_steps": 1000,
        "success_threshold": 200,
        "learning_rate": 5e-4,
        "hidden_size": [128, 128],
        "gamma": 0.99,
        "max_timesteps": 200000,
        "eval_freq": 2000,
        "eval_episodes": 10,
        "max_time": 7200,
    }
}

def play_episode(
    env: gym.Env,
    agent: Reinforce,
    train: bool = True,
    explore: bool = True,
    render: bool = False,
    max_steps: int = 200,
) -> Tuple[int, float, Dict]:
    """Play one episode and train reinforce algorithm"""
    ep_data = defaultdict(list)
    obs = env.reset()
    done = False
    num_steps = 0
    episode_return = 0

    observations = []
    actions = []
    rewards = []

    # Early rewards for Acrobot (reward shaping)
    is_acrobot = 'Acrobot' in str(env.observation_space)
    if is_acrobot:
        prev_obs = obs
        best_height = -1
        consecutive_good_actions = 0
        last_action = None
        action_repeat_count = 0

    while not done and num_steps < max_steps:
        if render:
            env.render()

        action = agent.act(np.array(obs), explore=explore)
        next_obs, reward, done, _ = env.step(action)

        # Enhanced reward shaping for Acrobot
        if is_acrobot:
            # Calculate the height of the end of the second link
            cos_theta1, sin_theta1 = next_obs[0], next_obs[1]
            cos_theta2, sin_theta2 = next_obs[2], next_obs[3]
            height = -cos_theta1 - cos_theta2
            
            # Calculate velocities and accelerations
            angular_vel1, angular_vel2 = next_obs[4], next_obs[5]
            prev_vel1 = prev_obs[4] if prev_obs is not None else 0
            prev_vel2 = prev_obs[5] if prev_obs is not None else 0
            accel1 = angular_vel1 - prev_vel1
            accel2 = angular_vel2 - prev_vel2

            # Progressive height rewards
            if height > best_height:
                height_improvement = height - best_height
                reward += 0.3 * height_improvement  # Increased reward for height
                best_height = height
                consecutive_good_actions += 1
            else:
                consecutive_good_actions = max(0, consecutive_good_actions - 1)

            # Momentum-based rewards
            if height > -1.5:  # When getting closer to goal
                # Reward for good momentum
                momentum_reward = 0.15 * (abs(angular_vel1) + abs(angular_vel2))
                # Additional reward for acceleration in right direction
                if height > 0:
                    momentum_reward *= 1.5  # Boost momentum reward when close to goal
                reward += momentum_reward

            # Position-based rewards
            if height > 0:  # Above horizontal
                reward += 0.2 * height  # Progressive reward for height
                if height > 1.0:  # Close to goal
                    reward += 1.0  # Bigger bonus near goal

            # Consistency rewards
            if consecutive_good_actions > 2:
                reward += 0.3 * consecutive_good_actions  # Increased bonus for consistency

            # Penalties
            if abs(angular_vel1) < 0.05 and abs(angular_vel2) < 0.05:
                reward -= 0.3  # Increased penalty for being too still

            # Action diversity encouragement
            if last_action == action:
                action_repeat_count += 1
                if action_repeat_count > 2:  # Reduced threshold
                    reward -= 0.2 * action_repeat_count
            else:
                action_repeat_count = 0
                reward += 0.1  # Small bonus for action change

            last_action = action
            prev_obs = next_obs.copy()  # Make sure to copy the array

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)

        num_steps += 1
        episode_return += reward
        obs = next_obs

    if train:
        update_info = agent.update(rewards, observations, actions)
        for k, v in update_info.items():
            ep_data[k].append(v)

    ep_data["episode_return"] = episode_return
    ep_data["episode_timesteps"] = num_steps

    return num_steps, episode_return, ep_data

def evaluate(env: gym.Env, agent: Reinforce, num_episodes: int, max_steps: int) -> Tuple[float, float]:
    """Evaluate the agent for a given number of episodes."""
    returns = []
    for _ in range(num_episodes):
        _, episode_return, _ = play_episode(
            env,
            agent,
            train=False,
            explore=False,
            render=False,
            max_steps=max_steps
        )
        returns.append(episode_return)

    return float(np.mean(returns)), float(np.std(returns))

def train(env: gym.Env, config: Dict, output: bool = True) -> Tuple[List[float], List[int], List[float], Dict]:
    """Training loop with environment-specific configurations and model saving."""
    timesteps_elapsed = 0
    max_steps = config["max_steps"]
    best_eval_return = -float('inf')
    best_model_path = f"best_{env.spec.id}_reinforce_model.pt"
    latest_model_path = f"latest_{env.spec.id}_reinforce_model.pt"
    patience = 5
    no_improvement_count = 0

    print("\nInitializing REINFORCE agent...")
    agent = Reinforce(
        action_space=env.action_space,
        observation_space=env.observation_space,
        **config,
    )

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
            episode_timesteps, episode_return, ep_data = play_episode(
                env,
                agent,
                train=True,
                explore=True,
                max_steps=max_steps
            )
            
            timesteps_elapsed += episode_timesteps
            episode_returns.append(episode_return)
            pbar.update(episode_timesteps)

            # Show progress
            if len(episode_returns) % 25 == 0:
                recent_returns = episode_returns[-25:]
                mean_return = np.mean(recent_returns)
                std_return = np.std(recent_returns)
                print(f"\nEpisode {len(episode_returns)}")
                print(f"Last 25 episodes - Mean: {mean_return:.2f}, Std: {std_return:.2f}")

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
    print(f"Total time elapsed: {(time.time() - start_time):.2f}s")
    print(f"Best model saved to: {best_model_path}")
    print(f"Latest model saved to: {latest_model_path}")

    return eval_returns, eval_timesteps, times, {}

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Train a REINFORCE agent on various gym environments')
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
