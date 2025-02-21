#!/usr/bin/env python
"""
Evaluate the DDPG agent for BipedalWalker-v3 using a configuration.
Runs a number of evaluation episodes (without exploration noise) and returns episode rewards.
Usage: evaluate(env, config)
"""

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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().squeeze(0)
        self.actor.train()
        return np.clip(action, -self.action_bound, self.action_bound)

def evaluate(env, config, eval_episodes=None):
    if eval_episodes is None:
        eval_episodes = config.get("eval_episodes", 5)
    state_dim = env.observation_space.shape[0]
    action_dim = env.observation_space.shape[0] if len(env.observation_space.shape)==1 else env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    agent = DDPGAgent(state_dim, action_dim, action_bound, device)
    checkpoint_path = config.get("save_filename", "bipedal_ddpg_checkpoint.pt")
    agent.load(checkpoint_path)
    returns = []
    for ep in range(eval_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.select_action(state)
            state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            ep_reward += reward
        returns.append(ep_reward)
        print(f"Episode {ep+1}: Reward = {ep_reward:.2f}")
    avg_return = np.mean(returns)
    print("Average Return:", avg_return)
    return returns

if __name__ == "__main__":
    sample_config = {
        "env": "BipedalWalker-v3",
        "eval_episodes": 5,
        "save_filename": "bipedal_ddpg_checkpoint.pt"
    }
    env = gym.make(sample_config["env"])
    evaluate(env, sample_config)
    env.close()
