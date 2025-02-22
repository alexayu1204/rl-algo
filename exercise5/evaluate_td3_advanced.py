#!/usr/bin/env python
"""
Evaluate an advanced TD3 checkpoint for BipedalWalker-v3.

Usage:
    python exercise5/evaluate_td3_advanced.py --checkpoint_path PATH --episodes NUM --render --env_mode MODE

--checkpoint_path: Path to the checkpoint file.
--episodes: Number of evaluation episodes.
--render: Render the environment during evaluation.
--env_mode: Render mode (e.g., "human" or "rgb_array").
"""

import argparse
import torch
import gym
import numpy as np
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
    
# Advanced FCNetwork (same as in training)
class FCNetwork(nn.Module):
    def __init__(self, dims, use_layer_norm=False, output_activation=None):
        super().__init__()
        layers = []
        self.use_layer_norm = use_layer_norm
        for i in range(len(dims)-1):
            linear = nn.Linear(dims[i], dims[i+1])
            layers.append(linear)
            if i < len(dims)-2:
                if self.use_layer_norm:
                    layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(nn.ReLU())
        if output_activation:
            layers.append(output_activation())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# The evaluation agent reconstructs the actor using the stored dimensions.
class TD3AgentAdvanced:
    def __init__(self, device, use_layer_norm=True):
        self.device = device
        self.actor = None
        self.action_bound = None
        self.use_layer_norm = use_layer_norm

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        actor_dims = checkpoint.get("actor_dims", None)
        if actor_dims is None:
            raise ValueError("Checkpoint missing 'actor_dims'!")
        # Use the stored dimensions to build the network.
        # Also use the stored "net_type" if available (default to base if not provided)
        net_type = checkpoint.get("net_type", "base")
        if net_type == "residual":
            from exercise5.train_td3_advanced import AdvancedActorResidual
            self.actor = AdvancedActorResidual(actor_dims[0], actor_dims[-1], hidden_dims=actor_dims[1:-1], use_layer_norm=self.use_layer_norm).to(self.device)
        elif net_type == "deep":
            from exercise5.train_td3_advanced import AdvancedActorDeep
            self.actor = AdvancedActorDeep(actor_dims[0], actor_dims[-1], hidden_dims=actor_dims[1:-1], use_layer_norm=self.use_layer_norm).to(self.device)
        else:
            self.actor = FCNetwork(actor_dims, use_layer_norm=self.use_layer_norm, output_activation=nn.Tanh).to(self.device)
        self.actor.load_state_dict(checkpoint["actor"])

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().squeeze(0)
        self.actor.train()
        return np.clip(action, -self.action_bound, self.action_bound)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an advanced TD3 checkpoint for BipedalWalker-v3")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument("--render", action="store_true", help="Render the environment during evaluation.")
    parser.add_argument("--env_mode", type=str, default=None, help='Render mode (e.g., "human", "rgb_array")')
    return parser.parse_args()

def evaluate(checkpoint_path, num_episodes=5, render=False, env_mode=None):
    env = gym.make("BipedalWalker-v3", render_mode=env_mode)
    # For evaluation, we use the actor_dims from the checkpoint. (Do not use current env dims.)
    action_bound = env.action_space.high[0]
    agent = TD3AgentAdvanced(device, use_layer_norm=True)
    agent.action_bound = action_bound
    agent.load(checkpoint_path)
    returns = []
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.select_action(state)
            state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            ep_reward += reward
            if render:
                env.render()
        returns.append(ep_reward)
        print(f"Episode {ep+1}: Reward = {ep_reward:.2f}")
    avg_return = np.mean(returns)
    print("Average Return:", avg_return)
    env.close()
    return returns

if __name__ == "__main__":
    args = parse_args()
    evaluate(args.checkpoint_path, num_episodes=args.episodes, render=args.render, env_mode=args.env_mode)
