#!/usr/bin/env python
"""
Enhanced DDPG training for BipedalWalker-v3 with:
  - Warmup period
  - Soft target updates
  - Gaussian exploration
  - Linear learning-rate decay
  - Linear noise decay

Usage: train(env, config)
"""

import os
import time
import math
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import gym
from collections import namedtuple
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

def linear_lr_schedule(optimizer, initial_lr, final_lr, current_step, max_step):
    """
    Linearly decay the optimizer's LR from initial_lr to final_lr over max_step steps.
    """
    if max_step <= 0:
        return
    new_lr = final_lr + (initial_lr - final_lr) * max(0, (max_step - current_step)) / max_step
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

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

    def hard_update(self, source):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.position = 0

    def add(self, state, action, next_state, reward, done):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.FloatTensor(action) if isinstance(action, (np.ndarray, list)) else torch.FloatTensor([action])
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([float(done)])
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(state, action, next_state, reward, done))
        else:
            self.buffer[self.position] = Transition(state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in indices]
        state = torch.stack([b.state for b in batch]).to(self.device)
        action = torch.stack([b.action for b in batch]).to(self.device)
        next_state = torch.stack([b.next_state for b in batch]).to(self.device)
        reward = torch.stack([b.reward for b in batch]).to(self.device)
        done = torch.stack([b.done for b in batch]).to(self.device)
        return Transition(state, action, next_state, reward, done)

    def __len__(self):
        return len(self.buffer)

class DiagGaussian:
    def __init__(self, action_dim, std=0.1):
        self.action_dim = action_dim
        self.std = std

    def sample(self):
        return np.random.normal(0, self.std, size=self.action_dim)

class DDPGAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        action_bound,
        gamma,
        tau,
        actor_lr_start,
        actor_lr_end,
        critic_lr_start,
        critic_lr_end,
        config,
        device,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_bound = action_bound

        policy_hidden_size = config.get("policy_hidden_size", [400, 300])
        critic_hidden_size = config.get("critic_hidden_size", [400, 300])

        # Actor and target
        self.actor = FCNetwork([state_dim, *policy_hidden_size, action_dim], output_activation=nn.Tanh).to(device)
        self.actor_target = FCNetwork([state_dim, *policy_hidden_size, action_dim], output_activation=nn.Tanh).to(device)
        self.actor_target.hard_update(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr_start)

        # Critic and target
        self.critic = FCNetwork([state_dim + action_dim, *critic_hidden_size, 1]).to(device)
        self.critic_target = FCNetwork([state_dim + action_dim, *critic_hidden_size, 1]).to(device)
        self.critic_target.hard_update(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr_start)

        self.actor_lr_start = actor_lr_start
        self.actor_lr_end = actor_lr_end
        self.critic_lr_start = critic_lr_start
        self.critic_lr_end = critic_lr_end

        # Noise scheduling
        self.noise_std_start = config.get("noise_std_start", 0.2)  # bigger initial noise
        self.noise_std_end = config.get("noise_std_end", 0.05)     # smaller final noise
        self.noise = DiagGaussian(action_dim, std=self.noise_std_start)

    def linear_noise_schedule(self, current_step, max_step):
        if max_step <= 0:
            return
        new_std = self.noise_std_end + (self.noise_std_start - self.noise_std_end) * max(0, (max_step - current_step)) / max_step
        self.noise.std = new_std

    def select_action(self, state, noise=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().squeeze(0)
        self.actor.train()
        if noise:
            action += self.noise.sample()
        return np.clip(action, -self.action_bound, self.action_bound)

    def update(self, batch):
        state, action, next_state, reward, done = batch

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(torch.cat([next_state, next_action], dim=1))
            target_value = reward + self.gamma * (1 - done) * target_q

        current_q = self.critic(torch.cat([state, action], dim=1))
        critic_loss = nn.MSELoss()(current_q, target_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(torch.cat([state, self.actor(state)], dim=1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss

def train(env, config):
    """
    Enhanced DDPG training with:
      - warmup_steps
      - linear LR scheduling
      - linear noise scheduling

    Keys needed in config:
      "actor_lr_start", "actor_lr_end"
      "critic_lr_start", "critic_lr_end"
      "noise_std_start", "noise_std_end"
      "policy_hidden_size", "critic_hidden_size"
      ...
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    max_timesteps = config.get("max_timesteps", 400000)
    episode_length = config.get("episode_length", 1600)
    batch_size = config["batch_size"]
    replay_capacity = config["buffer_capacity"]
    warmup_steps = config.get("warmup_steps", 1000)

    # Start vs. end LRs
    actor_lr_start = config.get("actor_lr_start", config["policy_learning_rate"])   # fallback
    actor_lr_end   = config.get("actor_lr_end", actor_lr_start/10)                  # default: 10x decay
    critic_lr_start= config.get("critic_lr_start", config["critic_learning_rate"])
    critic_lr_end  = config.get("critic_lr_end", critic_lr_start/10)

    # Noise schedule
    noise_std_start= config.get("noise_std_start", 0.2)
    noise_std_end  = config.get("noise_std_end", 0.05)

    gamma = config["gamma"]
    tau   = config["tau"]

    agent = DDPGAgent(
        state_dim,
        action_dim,
        action_bound,
        gamma,
        tau,
        actor_lr_start,
        actor_lr_end,
        critic_lr_start,
        critic_lr_end,
        config,
        device,
    )
    # Overwrite initial noise if user sets it in config
    agent.noise.std = noise_std_start

    replay_buffer = ReplayBuffer(replay_capacity, device)

    total_timesteps = 0
    episode_num = 0
    train_ep_returns = []
    pbar = tqdm(total=max_timesteps)
    start_time = time.time()

    while total_timesteps < max_timesteps:
        state, _ = env.reset()
        episode_reward = 0.0

        for t in range(episode_length):
            # Update LR linearly
            linear_lr_schedule(agent.actor_optimizer, actor_lr_start, actor_lr_end, total_timesteps, max_timesteps)
            linear_lr_schedule(agent.critic_optimizer, critic_lr_start, critic_lr_end, total_timesteps, max_timesteps)

            # Update noise linearly
            agent.linear_noise_schedule(total_timesteps, max_timesteps)

            if total_timesteps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, noise=True)

            next_state, reward, done, truncated, _ = env.step(action)
            done_flag = done or truncated

            replay_buffer.add(state, action, next_state, reward, done_flag)
            state = next_state
            episode_reward += reward
            total_timesteps += 1
            pbar.update(1)

            if total_timesteps >= warmup_steps and len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                agent.update(batch)

            if done_flag:
                break

        episode_num += 1
        train_ep_returns.append(episode_reward)
        pbar.write(f"Episode {episode_num}: Reward = {episode_reward:.2f}, Total Timesteps = {total_timesteps}")

    pbar.close()
    total_time = time.time() - start_time
    print(f"Training complete. Total timesteps: {total_timesteps}, Time: {total_time:.2f}s")

    checkpoint = {
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "actor_target": agent.actor_target.state_dict(),
        "critic_target": agent.critic_target.state_dict()
    }
    save_filename = config.get("save_filename", "bipedal_ddpg_checkpoint.pt")
    torch.save(checkpoint, save_filename)
    env.close()

    return {"train_ep_returns": train_ep_returns}, total_timesteps, total_time, []

if __name__ == "__main__":
    # Example usage with new scheduling keys:
    sample_config = {
        "env": "BipedalWalker-v3",
        "policy_learning_rate": 1e-4,
        "critic_learning_rate": 1e-3,
        "actor_lr_start": 1e-4,     # start actor LR
        "actor_lr_end":   1e-5,     # end actor LR
        "critic_lr_start":1e-3,
        "critic_lr_end":  1e-4,
        "noise_std_start":0.2,
        "noise_std_end":  0.05,
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 256,
        "buffer_capacity": int(1e6),
        "max_timesteps": 100000,
        "episode_length": 1600,
        "warmup_steps": 500,
        "save_filename": "bipedal_ddpg_checkpoint.pt",
        "algo": "DDPG",
        "policy_hidden_size": [400, 300],
        "critic_hidden_size": [400, 300],
    }
    env = gym.make(sample_config["env"])
    train(env, sample_config)
