import gym
import torch
import numpy as np
import time
import os
import argparse
from typing import List, Tuple, Dict

from exercise3.agents import Reinforce
from exercise3.train_reinforce import play_episode, ENV_CONFIGS

def evaluate_model(env: gym.Env, config: Dict, num_episodes: int, render: bool = False, model_path: str = None) -> List[float]:
    """Evaluate a trained model across multiple episodes."""
    agent = Reinforce(
        action_space=env.action_space,
        observation_space=env.observation_space,
        **config
    )
    
    if model_path is None:
        model_path = f"best_{env.spec.id}_reinforce_model.pt"
    
    try:
        agent.restore(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise ValueError(f"Could not find model to load at {model_path}")

    returns = []
    for episode in range(num_episodes):
        _, episode_return, _ = play_episode(
            env,
            agent,
            train=False,
            explore=False,
            render=render,
            max_steps=config["max_steps"]
        )
        returns.append(episode_return)
        print(f"Episode {episode + 1}/{num_episodes}: Return = {episode_return:.2f}")

    mean_return = np.mean(returns)
    std_return = np.std(returns)
    print(f"\nEvaluation Results over {num_episodes} episodes:")
    print(f"Mean Return: {mean_return:.2f} +/- {std_return:.2f}")
    print(f"Min Return: {min(returns):.2f}")
    print(f"Max Return: {max(returns):.2f}")
    
    return returns

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained REINFORCE agent on various gym environments')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        choices=list(ENV_CONFIGS.keys()),
                        help='Gym environment to evaluate')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to the saved model (default: best_[env_name]_reinforce_model.pt)')
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
    model_path = args.model_path or f"best_{args.env}_reinforce_model.pt"
    
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
