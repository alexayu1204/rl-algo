#!/usr/bin/env python
"""
Full DDPG Training Script with MPS (or CUDA/CPU) support for Apple Silicon.
This script defines a fully connected network, a replay buffer,
a DDPG agent, and the training loop with gradual logging.
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
from collections import namedtuple, defaultdict
from tqdm import tqdm

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# -------------------------
# Device Setup (MPS/CPU/CUDA)
# -------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

# -------------------------
# Fully Connected Network
# -------------------------
class FCNetwork(nn.Module):
    def __init__(self, dims, output_activation=None, hidden_activation=nn.ReLU, use_batch_norm=False):
        """
        dims: an iterable of layer sizes, e.g. (input_dim, hidden1, ..., output_dim)
        """
        super().__init__()
        self.input_size = dims[0]
        self.out_size = dims[-1]
        self.use_batch_norm = use_batch_norm
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(hidden_activation())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation:
            layers.append(output_activation())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_batch_norm and x.dim() == 1:
            x = x.unsqueeze(0)
        return self.layers(x)
    
    def hard_update(self, source):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)
    
    def soft_update(self, source, tau):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

# -------------------------
# Replay Buffer
# -------------------------
Transition = namedtuple("Transition", ("states", "actions", "next_states", "rewards", "done"))

class ReplayBuffer:
    def __init__(self, size, device):
        self.size = size
        self.device = device
        self.reset()
    
    def reset(self):
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.position = 0
        self.is_full = False
    
    def add(self, obs, action, next_obs, reward, done):
        # Convert observations to tensors
        state = torch.FloatTensor(obs)
        next_state = torch.FloatTensor(next_obs)
        # For continuous actions, action is expected to be a sequence
        if isinstance(action, (np.ndarray, list)):
            action_tensor = torch.FloatTensor(action)
        else:
            # For discrete actions, wrap it in a list
            action_tensor = torch.LongTensor([action])
        reward_tensor = torch.FloatTensor([reward])
        done_tensor = torch.FloatTensor([float(done)])
        
        if len(self.states) < self.size:
            self.states.append(state)
            self.actions.append(action_tensor)
            self.next_states.append(next_state)
            self.rewards.append(reward_tensor)
            self.dones.append(done_tensor)
        else:
            self.states[self.position] = state
            self.actions[self.position] = action_tensor
            self.next_states[self.position] = next_state
            self.rewards[self.position] = reward_tensor
            self.dones[self.position] = done_tensor
            self.is_full = True
        
        self.position = (self.position + 1) % self.size
    
    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.states), size=min(batch_size, len(self.states)))
        states = torch.stack([self.states[i] for i in indices]).to(self.device)
        actions = torch.stack([self.actions[i] for i in indices]).to(self.device)
        next_states = torch.stack([self.next_states[i] for i in indices]).to(self.device)
        rewards = torch.stack([self.rewards[i] for i in indices]).to(self.device)
        dones = torch.stack([self.dones[i] for i in indices]).to(self.device)
        return Transition(states, actions, next_states, rewards, dones)
    
    def __len__(self):
        return len(self.states)

# -------------------------
# DDPG Agent
# -------------------------
class DiagGaussian(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        # Register buffers so they move automatically with the device
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
    
    def sample(self):
        eps = torch.randn_like(self.mean)
        return self.mean + self.std * eps

class DDPGAgent:
    def __init__(self, action_space, observation_space, gamma, critic_learning_rate,
                 policy_learning_rate, critic_hidden_size, policy_hidden_size, tau, device):
        self.device = device
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.shape[0]
        
        self.upper_action_bound = action_space.high[0]
        self.lower_action_bound = action_space.low[0]
        
        # Actor network and its target
        self.actor = FCNetwork((STATE_SIZE, *policy_hidden_size, ACTION_SIZE),
                                 output_activation=torch.nn.Tanh).to(self.device)
        self.actor_target = FCNetwork((STATE_SIZE, *policy_hidden_size, ACTION_SIZE),
                                      output_activation=torch.nn.Tanh).to(self.device)
        self.actor_target.hard_update(self.actor)
        
        # Critic network and its target
        self.critic = FCNetwork((STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1),
                                  output_activation=None).to(self.device)
        self.critic_target = FCNetwork((STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1),
                                       output_activation=None).to(self.device)
        self.critic_target.hard_update(self.critic)
        
        # Optimizers
        self.policy_optim = Adam(self.actor.parameters(), lr=policy_learning_rate, eps=1e-3)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_learning_rate, eps=1e-3)
        
        self.gamma = gamma
        self.tau = tau
        
        # Save initial learning rates for scheduling
        self.lr1 = policy_learning_rate
        self.lr2 = critic_learning_rate
        self.lr_decay_strategy = "linear"  # Options: "constant", "linear", "exponential"
        self.lr_min = 1e-6
        self.lr_exponential_decay_factor = 0.01
        self.learning_rate_fraction = 0.2
        
        # Define Gaussian noise for exploration
        mean = torch.zeros(ACTION_SIZE).to(self.device)
        std = 0.1 * torch.ones(ACTION_SIZE).to(self.device)
        self.noise = DiagGaussian(mean, std)
    
    def schedule_hyperparameters(self, timestep, max_timesteps):
        def lr_linear_decay(timestep, max_timestep, lr_start, lr_min, fraction):
            frac = min(1.0, timestep / (max_timestep * fraction))
            return lr_start - frac * (lr_start - lr_min)
        def lr_exponential_decay(timestep, lr_start, lr_min, decay):
            return lr_min + (lr_start - lr_min) * math.exp(-decay * timestep)
        if self.lr_decay_strategy == "constant":
            new_policy_lr = self.lr1
            new_critic_lr = self.lr2
        elif self.lr_decay_strategy == "linear":
            new_policy_lr = lr_linear_decay(timestep, max_timesteps, self.lr1, self.lr_min, self.learning_rate_fraction)
            new_critic_lr = lr_linear_decay(timestep, max_timesteps, self.lr2, self.lr_min, self.learning_rate_fraction)
        elif self.lr_decay_strategy == "exponential":
            new_policy_lr = lr_exponential_decay(timestep, self.lr1, self.lr_min, self.lr_exponential_decay_factor)
            new_critic_lr = lr_exponential_decay(timestep, self.lr2, self.lr_min, self.lr_exponential_decay_factor)
        else:
            raise ValueError("Invalid lr_decay_strategy")
        for param_group in self.policy_optim.param_groups:
            param_group['lr'] = new_policy_lr
        for param_group in self.critic_optim.param_groups:
            param_group['lr'] = new_critic_lr
    
    def act(self, obs, explore=True):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs_tensor).squeeze(0)
        if explore:
            noise_sample = self.noise.sample().to(self.device)
            action = action + noise_sample
        action = torch.clamp(action, self.lower_action_bound, self.upper_action_bound)
        return action.cpu().numpy()
    
    def update(self, batch):
        states, actions, next_states, rewards, dones = batch
        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q_values = self.critic_target(torch.cat([next_states, next_actions], dim=-1)).detach()
            target_values = rewards + self.gamma * (1 - dones) * target_q_values
        predicted_q_values = self.critic(torch.cat([states, actions], dim=-1))
        q_loss = F.mse_loss(predicted_q_values, target_values)
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()
        
        # Actor update
        predicted_actions = self.actor(states)
        policy_loss = -self.critic(torch.cat([states, predicted_actions], dim=-1)).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        # Soft update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {"q_loss": q_loss.item(), "p_loss": policy_loss.item()}

# -------------------------
# Training Loop
# -------------------------
def play_episode(env, agent, replay_buffer, train=True, explore=True, render=False, max_steps=200, batch_size=64):
    ep_data = defaultdict(list)
    # Use Gym v0.26+ API: reset returns (obs, info)
    obs, _ = env.reset()
    done = False
    episode_timesteps = 0
    episode_return = 0
    while not done:
        action = agent.act(obs, explore=explore)
        # Step returns: (obs, reward, done, truncated, info)
        nobs, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        if train:
            replay_buffer.add(np.array(obs, dtype=np.float32),
                              np.array(action, dtype=np.float32),
                              np.array(nobs, dtype=np.float32),
                              float(reward),
                              bool(done))
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                new_data = agent.update(batch)
                for k, v in new_data.items():
                    ep_data[k].append(v)
        episode_timesteps += 1
        episode_return += reward
        if render:
            env.render()
        if episode_timesteps >= max_steps:
            break
        obs = nobs
    return episode_timesteps, episode_return, ep_data

def train(env, config, device):
    timesteps_elapsed = 0
    max_steps = config["episode_length"]
    max_timesteps = config["max_timesteps"]
    best_eval_return = -float('inf')
    
    agent = DDPGAgent(
        env.action_space,
        env.observation_space,
        gamma=config["gamma"],
        critic_learning_rate=config["critic_learning_rate"],
        policy_learning_rate=config["policy_learning_rate"],
        critic_hidden_size=config["critic_hidden_size"],
        policy_hidden_size=config["policy_hidden_size"],
        tau=config["tau"],
        device=device
    )
    replay_buffer = ReplayBuffer(config["buffer_capacity"], device)
    
    eval_returns_all = []
    start_time = time.time()
    episode_count = 0
    pbar = tqdm(total=max_timesteps)
    
    while timesteps_elapsed < max_timesteps:
        agent.schedule_hyperparameters(timesteps_elapsed, max_timesteps)
        episode_timesteps, ep_return, ep_data = play_episode(
            env, agent, replay_buffer, train=True, explore=True,
            render=False, max_steps=max_steps, batch_size=config["batch_size"]
        )
        episode_count += 1
        timesteps_elapsed += episode_timesteps
        pbar.update(episode_timesteps)
        
        # Log average losses if available
        if len(ep_data["q_loss"]) > 0:
            avg_q_loss = sum(ep_data["q_loss"]) / len(ep_data["q_loss"])
            avg_p_loss = sum(ep_data["p_loss"]) / len(ep_data["p_loss"])
        else:
            avg_q_loss, avg_p_loss = 0, 0
        if episode_count % 10 == 0:
            print(f"Episode: {episode_count}, timesteps: {timesteps_elapsed}, return: {ep_return:.2f}, avg_q_loss: {avg_q_loss:.4f}, avg_p_loss: {avg_p_loss:.4f}")
        
        # Evaluation phase (every eval_freq timesteps)
        if timesteps_elapsed % config["eval_freq"] < episode_timesteps:
            eval_returns = 0
            for _ in range(config["eval_episodes"]):
                _, episode_return, _ = play_episode(
                    env, agent, replay_buffer, train=False, explore=False,
                    render=False, max_steps=max_steps, batch_size=config["batch_size"]
                )
                eval_returns += episode_return / config["eval_episodes"]
            eval_returns_all.append(eval_returns)
            print(f"\nEvaluation at timestep {timesteps_elapsed}: Mean Return: {eval_returns:.2f}")
            if eval_returns >= config["target_return"]:
                print(f"Target achieved: {eval_returns:.2f} >= {config['target_return']}")
                break
    pbar.close()
    total_time = time.time() - start_time
    print(f"Training complete. Total timesteps: {timesteps_elapsed}, Time: {total_time:.2f}s")
    return eval_returns_all

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # Updated configuration with the missing keys for hidden sizes.
    config = {
        "env": "BipedalWalker-v3",
        "eval_freq": 20000,
        "eval_episodes": 10,  # Reduced for quicker evaluations during testing
        "policy_learning_rate": 1e-4,
        "critic_learning_rate": 1e-3,
        "target_return": 300.0,
        "episode_length": 1600,
        "max_timesteps": 400000,
        "max_time": 120 * 60,
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 64,
        "buffer_capacity": int(1e6),
        "critic_hidden_size": [400, 300],
        "policy_hidden_size": [400, 300],
    }
    env = gym.make(config["env"])
    eval_returns = train(env, config, device)
    env.close()
    print("Mean returns:", np.mean(eval_returns))
    # plot the evaluation returns
    import matplotlib.pyplot as plt
    plt.plot(eval_returns)
    plt.xlabel("Evaluation")
    plt.ylabel("Return")
    plt.show()

