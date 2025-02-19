import gym
import torch
import numpy as np
import time
import os
import argparse
from typing import List, Tuple, Dict

from exercise3.agents import DQN

# Default configurations for different environments
ENV_CONFIGS = {
    "CartPole-v1": {
        "max_steps": 500,
        "success_threshold": 295,
        "render_delay": 0.02,
        "hidden_size": [256, 256],
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "target_update_freq": 10,
        "batch_size": 256,
        "buffer_capacity": int(1e5),
        "epsilon_decay": 0.995,
        "tau": 0.01,
        "epsilon_start": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay_strategy": "exponential",
    },
    "Acrobot-v1": {
        "max_steps": 500,
        "success_threshold": -100,  # Acrobot is solved when average reward is > -100
        "render_delay": 0.02,
        "hidden_size": [64, 64],
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "target_update_freq": 5,
        "batch_size": 64,
        "buffer_capacity": int(1e5),
        "epsilon_decay": 0.995,
        "tau": 0.01,
        "epsilon_start": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay_strategy": "exponential",
    },
    "BipedalWalker-v3": {
        "max_steps": 1600,
        "success_threshold": 300,  # BipedalWalker is solved when average reward is > 300
        "render_delay": 0.03,
        "hidden_size": [256, 256],
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "target_update_freq": 5,
        "batch_size": 128,
        "buffer_capacity": int(1e6),
        "epsilon_decay": 0.999,
    },
    "LunarLander-v2": {
        "max_steps": 1000,
        "success_threshold": 200,
        "render_delay": 0.02,
        "hidden_size": [256, 256],
        "learning_rate": 5e-4,
        "gamma": 0.99,
        "target_update_freq": 10,
        "batch_size": 256,
        "buffer_capacity": int(2e5),
        "epsilon_decay": 0.997,
        "tau": 0.005,
        "epsilon_start": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay_strategy": "exponential",
    }
}

def evaluate_model(env: gym.Env, config: Dict, num_episodes: int, render: bool = False, model_path: str = None) -> List[float]:
    """Evaluate a trained model across multiple episodes."""
    agent = DQN(
        action_space=env.action_space,
        observation_space=env.observation_space,
        **config
    )
    
    if model_path is None:
        model_path = f"best_{env.spec.id}_model.pt"
    
    try:
        agent.restore(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise ValueError(f"Could not find model to load at {model_path}")

    returns = []
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_return = 0
        steps = 0

        while not done and steps < config["max_steps"]:
            if render:
                env.render()
                time.sleep(config.get("render_delay", 0.02))
            
            action = agent.act(obs, explore=False)
            obs, reward, done, _ = env.step(action)
            episode_return += reward
            steps += 1

        returns.append(episode_return)
        print(f"Episode {episode + 1}/{num_episodes}: Return = {episode_return:.2f}, Steps = {steps}")

    mean_return = np.mean(returns)
    std_return = np.std(returns)
    print(f"\nEvaluation Results over {num_episodes} episodes:")
    print(f"Mean Return: {mean_return:.2f} +/- {std_return:.2f}")
    print(f"Min Return: {min(returns):.2f}")
    print(f"Max Return: {max(returns):.2f}")
    
    return returns

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained DQN agent on various gym environments')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        choices=list(ENV_CONFIGS.keys()),
                        help='Gym environment to evaluate')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to the saved model (default: best_[env_name]_model.pt)')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Number of episodes to evaluate')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    # Get environment-specific configuration
    if args.env not in ENV_CONFIGS:
        raise ValueError(f"Environment {args.env} not supported. Choose from: {list(ENV_CONFIGS.keys())}")
    
    config = ENV_CONFIGS[args.env].copy()
    model_path = args.model_path or f"best_{args.env}_model.pt"
    
    print(f"Evaluating environment: {args.env}")
    print(f"Looking for model at: {model_path}")
    
    # Set up environment and seeds
    env = gym.make(args.env)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Run evaluation
    returns = evaluate_model(
        env, 
        config, 
        args.episodes,
        render=not args.no_render,
        model_path=model_path
    )
    env.close()
    
    # Print results
    print("\nOverall Results:")
    print(f"Average Return: {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    success_threshold = config["success_threshold"]
    success_rate = 100 * np.mean(np.array(returns) >= success_threshold)
    print(f"Success Rate (episodes with return >= {success_threshold}): {success_rate:.1f}%")
    
    if success_rate >= 95:
        print("\nExcellent performance! The agent has mastered the environment!")
    elif success_rate >= 80:
        print("\nGood performance! The agent has learned a strong policy.")
    else:
        print("\nThe agent's performance could be improved further.")
