#!/usr/bin/env python
"""
Tuned Baseline DDPG Training for BipedalWalker-v3 incorporating TD3-inspired improvements:
  - Target policy smoothing
  - Delayed actor updates
  - Linear noise decay (from 0.2 to 0.05)
  - Reduced total training timesteps (200000 instead of 400000)
  - Periodic evaluation and checkpointing (best and latest models saved)

Usage: python train_ddpg.py
"""

import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import gym
from collections import namedtuple
from tqdm import tqdm
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# Device selection: use CUDA if available, otherwise CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----- Fully Connected Network -----
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

# ----- Replay Buffer -----
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

# ----- Diagonal Gaussian Noise with decay -----
class DiagGaussian:
    def __init__(self, action_dim, initial_std):
        self.action_dim = action_dim
        self.std = initial_std
    def sample(self):
        return np.random.normal(0, self.std, size=self.action_dim)

# ----- DDPG Agent with TD3-inspired improvements -----
class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_bound, gamma, tau, actor_lr, critic_lr,
                 initial_noise_std, final_noise_std, max_timesteps, device, actor_update_delay=2):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_bound = action_bound

        # Actor network and target.
        self.actor = FCNetwork([state_dim, 400, 300, action_dim], output_activation=nn.Tanh).to(device)
        self.actor_target = FCNetwork([state_dim, 400, 300, action_dim], output_activation=nn.Tanh).to(device)
        self.actor_target.hard_update(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)

        # Critic network and target.
        self.critic = FCNetwork([state_dim + action_dim, 400, 300, 1]).to(device)
        self.critic_target = FCNetwork([state_dim + action_dim, 400, 300, 1]).to(device)
        self.critic_target.hard_update(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

        # Noise parameters.
        self.initial_noise_std = initial_noise_std
        self.final_noise_std = final_noise_std
        self.max_timesteps = max_timesteps
        self.noise = DiagGaussian(action_dim, initial_std=initial_noise_std)
        self.actor_update_delay = actor_update_delay
        self.update_count = 0

    def update_noise_std(self, current_timestep):
        frac = min(1.0, current_timestep / self.max_timesteps)
        new_std = self.initial_noise_std - frac * (self.initial_noise_std - self.final_noise_std)
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
        # Target Policy Smoothing: add clipped noise to target action.
        noise = torch.clamp(torch.randn_like(action) * 0.2, -0.5, 0.5)
        next_action = self.actor_target(next_state) + noise
        next_action = torch.clamp(next_action, -self.action_bound, self.action_bound)
        with torch.no_grad():
            target_q = self.critic_target(torch.cat([next_state, next_action], dim=1))
            target_value = reward + self.gamma * (1 - done) * target_q
        current_q = self.critic(torch.cat([state, action], dim=1))
        critic_loss = F.mse_loss(current_q, target_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed Actor Update.
        self.update_count += 1
        actor_loss = 0.0
        if self.update_count % self.actor_update_delay == 0:
            actor_loss = -self.critic(torch.cat([state, self.actor(state)], dim=1)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # Soft update target networks.
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return critic_loss.item(), actor_loss

# ----- Evaluation function -----
def evaluate_agent(env, agent, eval_episodes=5):
    total_reward = 0.0
    for _ in range(eval_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.select_action(state, noise=False)
            state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            ep_reward += reward
        total_reward += ep_reward
    return total_reward / eval_episodes

# ----- Training Loop -----
def train():
    env = gym.make("BipedalWalker-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    # Hyperparameters (tuned for faster convergence):
    gamma = 0.99
    tau = 0.005
    actor_lr = 1e-4
    critic_lr = 1e-3
    max_timesteps = 400000  # reduced from 400k to 200k
    episode_length = 1600
    batch_size = 64
    replay_capacity = int(1e6)
    warmup_steps = 1000

    # Noise decay: from 0.2 to 0.05.
    initial_noise_std = 0.2
    final_noise_std = 0.05

    agent = DDPGAgent(state_dim, action_dim, action_bound, gamma, tau, actor_lr, critic_lr,
                      initial_noise_std, final_noise_std, max_timesteps, device, actor_update_delay=2)
    replay_buffer = ReplayBuffer(replay_capacity, device)

    total_timesteps = 0
    episode_num = 0
    best_eval_reward = -float("inf")
    latest_checkpoint_path = "bipedal_ddpg_latest.pt"
    best_checkpoint_path = "bipedal_ddpg_best.pt"
    eval_freq = 20000

    pbar = tqdm(total=max_timesteps)
    start_time = time.time()
    while total_timesteps < max_timesteps:
        state, _ = env.reset()
        episode_reward = 0.0
        episode_timesteps = 0
        while episode_timesteps < episode_length:
            agent.update_noise_std(total_timesteps)
            if total_timesteps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, noise=True)
            next_state, reward, done, truncated, _ = env.step(action)
            done_flag = done or truncated
            replay_buffer.add(state, action, next_state, reward, done_flag)
            state = next_state
            episode_reward += reward
            episode_timesteps += 1
            total_timesteps += 1
            pbar.update(1)
            if total_timesteps >= warmup_steps and len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                agent.update(batch)
            if done_flag:
                break
        episode_num += 1
        pbar.write(f"Episode {episode_num}: Reward = {episode_reward:.2f}, Total Timesteps = {total_timesteps}")
        # Periodic evaluation.
        if total_timesteps % eval_freq < episode_timesteps:
            eval_reward = evaluate_agent(env, agent, eval_episodes=5)
            pbar.write(f"*** Evaluation at timestep {total_timesteps}: Avg Reward = {eval_reward:.2f} ***")
            checkpoint = {
                "actor": agent.actor.state_dict(),
                "critic": agent.critic.state_dict(),
                "actor_target": agent.actor_target.state_dict(),
                "critic_target": agent.critic_target.state_dict(),
            }
            torch.save(checkpoint, latest_checkpoint_path)
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save(checkpoint, best_checkpoint_path)
                pbar.write(f"*** New best model saved with Avg Reward = {best_eval_reward:.2f} ***")
    pbar.close()
    total_time = time.time() - start_time
    print(f"Training complete. Total timesteps: {total_timesteps}, Time: {total_time:.2f}s")
    env.close()

if __name__ == "__main__":
    train()
