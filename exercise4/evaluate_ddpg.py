#!/usr/bin/env python
"""
Evaluate a saved DDPG checkpoint for BipedalWalker-v3.
This script creates the actor network (with the same architecture as training),
loads its state_dict from the checkpoint, and then runs evaluation episodes.
"""

import torch
import gym
import numpy as np
import torch.nn as nn
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# --- Device Selection (CUDA if available, else CPU) ---
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

# --- Define the same actor network architecture ---
class FCNetwork(nn.Module):
    def __init__(self, dims, output_activation=None):
        super().__init__()
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
        if output_activation:
            layers.append(output_activation())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_bound, device):
        self.device = device
        self.action_bound = action_bound
        self.actor = FCNetwork([state_dim, 400, 300, action_dim], output_activation=nn.Tanh).to(device)
    def load(self, checkpoint_path):
        # Load the checkpoint dictionary and update the actor's state_dict.
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().squeeze(0)
        self.actor.train()
        return np.clip(action, -self.action_bound, self.action_bound)

def evaluate(checkpoint_path, num_episodes=5, render=False):
    env = gym.make("BipedalWalker-v3", render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    agent = DDPGAgent(state_dim, action_dim, action_bound, device)
    agent.load(checkpoint_path)
    returns = []
    for ep in range(num_episodes):
        state, _ = env.reset()
        ep_return = 0
        done = False
        while not done:
            if render:
                env.render()
            action = agent.select_action(state)
            state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            ep_return += reward
        returns.append(ep_return)
        print(f"Episode {ep+1}: Return = {ep_return:.2f}")
    env.close()
    avg_return = np.mean(returns)
    print("Average Return:", avg_return)
    return returns

if __name__ == "__main__":
    checkpoint_path = "bipedal_ddpg_checkpoint.pt"
    evaluate(checkpoint_path, num_episodes=5, render=True)
