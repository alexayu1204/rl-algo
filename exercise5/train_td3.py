#!/usr/bin/env python
"""
Enhanced TD3 training for BipedalWalker-v3 with Reward Normalization and Layer Normalization.

This script implements TD3 with:
  - Twin critics (clipped double-Q learning)
  - Delayed actor updates (policy_delay)
  - Target policy smoothing (with noise clipping)
  - Linear learning-rate decay for actor and critic optimizers
  - Linear noise decay for exploration
  - Reward normalization using running statistics
  - Improved network architecture with optional layer normalization and orthogonal initialization

Usage: train_td3(env, config)
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
    new_lr = final_lr + (initial_lr - final_lr) * max(0, (max_step - current_step)) / max_step
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

# Optional orthogonal initialization function.
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=1.414)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class FCNetwork(nn.Module):
    def __init__(self, dims, use_layer_norm=False, output_activation=None):
        super().__init__()
        layers = []
        self.use_layer_norm = use_layer_norm
        for i in range(len(dims)-1):
            linear = nn.Linear(dims[i], dims[i+1])
            linear.apply(init_weights)
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
        self.std = std
        self.action_dim = action_dim
    def sample(self):
        return np.random.normal(0, self.std, size=self.action_dim)

# Reward normalization using running mean and standard deviation.
class RewardNormalizer:
    def __init__(self, alpha=0.001):
        self.alpha = alpha
        self.running_mean = None
        self.running_std = None
    def normalize(self, reward):
        if self.running_mean is None:
            self.running_mean = reward
            self.running_std = 1.0
            return reward
        else:
            self.running_mean = (1 - self.alpha) * self.running_mean + self.alpha * reward
            self.running_std = (1 - self.alpha) * self.running_std + self.alpha * abs(reward - self.running_mean)
            norm_reward = (reward - self.running_mean) / (self.running_std + 1e-8)
            return np.clip(norm_reward, -10, 10)

class TD3Agent:
    def __init__(self, state_dim, action_dim, action_bound, gamma, tau,
                 actor_lr_start, actor_lr_end, critic_lr_start, critic_lr_end,
                 config, device, policy_delay=2, noise_clip=0.5, use_layer_norm=True):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_bound = action_bound
        self.policy_delay = policy_delay
        self.update_counter = 0

        policy_hidden_size = config.get("policy_hidden_size", [400,300])
        critic_hidden_size = config.get("critic_hidden_size", [400,300])

        # Actor and its target with LayerNorm if desired.
        self.actor = FCNetwork([state_dim, *policy_hidden_size, action_dim], 
                                use_layer_norm=use_layer_norm, output_activation=nn.Tanh).to(device)
        self.actor_target = FCNetwork([state_dim, *policy_hidden_size, action_dim],
                                       use_layer_norm=use_layer_norm, output_activation=nn.Tanh).to(device)
        self.actor_target.hard_update(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr_start)

        # Twin critics and their targets.
        self.critic1 = FCNetwork([state_dim + action_dim, *critic_hidden_size, 1],
                                  use_layer_norm=use_layer_norm).to(device)
        self.critic1_target = FCNetwork([state_dim + action_dim, *critic_hidden_size, 1],
                                         use_layer_norm=use_layer_norm).to(device)
        self.critic1_target.hard_update(self.critic1)
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=critic_lr_start)

        self.critic2 = FCNetwork([state_dim + action_dim, *critic_hidden_size, 1],
                                  use_layer_norm=use_layer_norm).to(device)
        self.critic2_target = FCNetwork([state_dim + action_dim, *critic_hidden_size, 1],
                                         use_layer_norm=use_layer_norm).to(device)
        self.critic2_target.hard_update(self.critic2)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=critic_lr_start)

        self.actor_lr_start = actor_lr_start
        self.actor_lr_end = actor_lr_end
        self.critic_lr_start = critic_lr_start
        self.critic_lr_end = critic_lr_end

        self.noise_std_start = config.get("noise_std_start", 0.2)
        self.noise_std_end = config.get("noise_std_end", 0.1)
        self.noise = DiagGaussian(action_dim, std=self.noise_std_start)
        self.noise_clip = noise_clip

    def linear_noise_schedule(self, current_step, max_step):
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

        # Target policy smoothing: add clipped noise
        noise = torch.randn_like(action) * self.noise.std
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        next_action = self.actor_target(next_state) + noise
        next_action = next_action.clamp(-self.action_bound, self.action_bound)

        with torch.no_grad():
            q1_target = self.critic1_target(torch.cat([next_state, next_action], dim=1))
            q2_target = self.critic2_target(torch.cat([next_state, next_action], dim=1))
            target_q = torch.min(q1_target, q2_target)
            target_value = reward + self.gamma * (1 - done) * target_q

        q1_current = self.critic1(torch.cat([state, action], dim=1))
        q2_current = self.critic2(torch.cat([state, action], dim=1))
        critic1_loss = nn.MSELoss()(q1_current, target_value)
        critic2_loss = nn.MSELoss()(q2_current, target_value)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        self.update_counter += 1
        actor_loss = 0.0
        if self.update_counter % self.policy_delay == 0:
            actor_loss = -self.critic1(torch.cat([state, self.actor(state)], dim=1)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return (critic1_loss.item() + critic2_loss.item()), actor_loss

class RewardNormalizer:
    def __init__(self, alpha=0.001):
        self.alpha = alpha
        self.running_mean = None
        self.running_std = None
    def normalize(self, reward):
        if self.running_mean is None:
            self.running_mean = reward
            self.running_std = 1.0
            return reward
        self.running_mean = (1 - self.alpha) * self.running_mean + self.alpha * reward
        self.running_std = (1 - self.alpha) * self.running_std + self.alpha * abs(reward - self.running_mean)
        norm_reward = (reward - self.running_mean) / (self.running_std + 1e-8)
        return np.clip(norm_reward, -10, 10)

def train(env, config):
    """
    Train TD3 on the given environment using parameters from config.
    Expected keys:
      - actor_lr_start, actor_lr_end, critic_lr_start, critic_lr_end,
      - policy_learning_rate, critic_learning_rate, gamma, tau, batch_size,
      - buffer_capacity, max_timesteps, episode_length, warmup_steps, reward_scale,
      - save_filename, algo, policy_hidden_size, critic_hidden_size,
      - noise_std_start, noise_std_end, policy_delay, noise_clip, use_layer_norm.
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    gamma = config["gamma"]
    tau = config["tau"]
    actor_lr_start = config.get("actor_lr_start", config["policy_learning_rate"])
    actor_lr_end   = config.get("actor_lr_end", actor_lr_start/10)
    critic_lr_start= config.get("critic_lr_start", config["critic_learning_rate"])
    critic_lr_end  = config.get("critic_lr_end", critic_lr_start/10)

    max_timesteps = config.get("max_timesteps", 400000)
    episode_length = config.get("episode_length", 1600)
    batch_size = config["batch_size"]
    replay_capacity = config["buffer_capacity"]
    warmup_steps = config.get("warmup_steps", 1000)
    reward_scale = config.get("reward_scale", 1.0)

    noise_std_start = config.get("noise_std_start", 0.2)
    noise_std_end = config.get("noise_std_end", 0.1)
    policy_delay = config.get("policy_delay", 2)
    noise_clip = config.get("noise_clip", 0.5)
    use_layer_norm = config.get("use_layer_norm", True)

    # Create a reward normalizer.
    reward_normalizer = RewardNormalizer(alpha=0.001)

    agent = TD3Agent(
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
        policy_delay=policy_delay,
        noise_clip=noise_clip,
        use_layer_norm=use_layer_norm,
    )
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
            # Update learning rates.
            linear_lr_schedule(agent.actor_optimizer, actor_lr_start, actor_lr_end, total_timesteps, max_timesteps)
            linear_lr_schedule(agent.critic1_optimizer, critic_lr_start, critic_lr_end, total_timesteps, max_timesteps)
            linear_lr_schedule(agent.critic2_optimizer, critic_lr_start, critic_lr_end, total_timesteps, max_timesteps)

            # Update noise schedule.
            agent.linear_noise_schedule(total_timesteps, max_timesteps)

            if total_timesteps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, noise=True)
            next_state, reward, done, truncated, _ = env.step(action)
            done_flag = done or truncated

            # Apply reward scaling and normalization.
            scaled_reward = reward * reward_scale
            norm_reward = reward_normalizer.normalize(scaled_reward)

            replay_buffer.add(state, action, next_state, norm_reward, done_flag)
            state = next_state
            episode_reward += reward  # log the original reward for monitoring
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

    # Save checkpoint along with actor dimensions for evaluation.
    actor_dims = [state_dim] + config.get("policy_hidden_size", [400,300]) + [action_dim]
    checkpoint = {
        "actor": agent.actor.state_dict(),
        "critic1": agent.critic1.state_dict(),
        "critic2": agent.critic2.state_dict(),
        "actor_target": agent.actor_target.state_dict(),
        "critic1_target": agent.critic1_target.state_dict(),
        "critic2_target": agent.critic2_target.state_dict(),
        "actor_dims": actor_dims,
    }
    save_filename = config.get("save_filename", "bipedal_td3_checkpoint.pt")
    torch.save(checkpoint, save_filename)
    env.close()
    return {"train_ep_returns": train_ep_returns}, total_timesteps, total_time, []

if __name__ == "__main__":
    sample_config = {
        "env": "BipedalWalker-v3",
        "policy_learning_rate": 1e-4,
        "critic_learning_rate": 1e-3,
        "actor_lr_start": 1e-4,
        "actor_lr_end": 1e-5,
        "critic_lr_start": 1e-3,
        "critic_lr_end": 1e-4,
        "noise_std_start": 0.2,
        "noise_std_end": 0.1,
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 128,
        "buffer_capacity": int(1e6),
        "max_timesteps": 250000,
        "episode_length": 1600,
        "warmup_steps": 500,
        "reward_scale": 0.1,  # Adjust as needed; try 0.1 to 1.0
        "save_filename": "bipedal_td3_checkpoint.pt",
        "algo": "TD3",
        "policy_hidden_size": [400,300],
        "critic_hidden_size": [400,300],
        "policy_delay": 2,
        "noise_clip": 0.5,
        "use_layer_norm": True,
    }
    env = gym.make(sample_config["env"])
    train(env, sample_config)
