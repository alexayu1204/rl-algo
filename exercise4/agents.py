import os
import gym
import numpy as np
from torch.optim import Adam
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal

from exercise3.agents import Agent
from exercise3.networks import FCNetwork
from exercise3.replay import Transition

# --- Revised DiagGaussian for clarity ---
class DiagGaussian(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        # Ensure mean and std are registered buffers so they’re on the correct device.
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def sample(self):
        # Sample noise on the same device as mean
        eps = torch.randn_like(self.mean)
        return self.mean + self.std * eps

class DDPG(Agent):
    """DDPG implementation for continuous control environments."""
    def __init__(
            self,
            action_space: gym.Space,
            observation_space: gym.Space,
            gamma: float,
            critic_learning_rate: float,
            policy_learning_rate: float,
            critic_hidden_size: list,
            policy_hidden_size: list,
            tau: float,
            **kwargs,
    ):
        super().__init__(action_space, observation_space)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.shape[0]

        self.upper_action_bound = action_space.high[0]
        self.lower_action_bound = action_space.low[0]

        # Create actor network and its target with Tanh output
        self.actor = FCNetwork((STATE_SIZE, *policy_hidden_size, ACTION_SIZE), output_activation=torch.nn.Tanh)
        self.actor_target = FCNetwork((STATE_SIZE, *policy_hidden_size, ACTION_SIZE), output_activation=torch.nn.Tanh)
        self.actor_target.hard_update(self.actor)

        # Create critic network and its target
        self.critic = FCNetwork((STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None)
        self.critic_target = FCNetwork((STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None)
        self.critic_target.hard_update(self.critic)

        # Move networks to device
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        # Create optimizers
        self.policy_optim = Adam(self.actor.parameters(), lr=policy_learning_rate, eps=1e-3)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_learning_rate, eps=1e-3)

        self.gamma = gamma
        self.tau = tau

        # Save initial learning rates for scheduling
        self.lr1 = policy_learning_rate
        self.lr2 = critic_learning_rate
        self.lr_decay_strategy = "linear"  # Could also be "exponential"
        self.lr_min = 1e-6
        self.lr_exponential_decay_factor = 0.01
        self.learning_rate_fraction = 0.2

        # Define a Gaussian noise model for exploration (make sure it is on the proper device)
        mean = torch.zeros(ACTION_SIZE).to(self.device)
        std = 0.1 * torch.ones(ACTION_SIZE).to(self.device)
        self.noise = DiagGaussian(mean, std)

        self.saveables.update({
            "actor": self.actor,
            "actor_target": self.actor_target,
            "critic": self.critic,
            "critic_target": self.critic_target,
            "policy_optim": self.policy_optim,
            "critic_optim": self.critic_optim,
        })

    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):
        """Update learning rates and other hyperparameters based on the timestep."""
        import math

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
            raise ValueError("Invalid lr_decay_strategy; choose 'constant', 'linear', or 'exponential'.")

        self.policy_learning_rate = new_policy_lr
        self.critic_learning_rate = new_critic_lr

        # Update the learning rates in the optimizer parameter groups
        for param_group in self.policy_optim.param_groups:
            param_group['lr'] = new_policy_lr
        for param_group in self.critic_optim.param_groups:
            param_group['lr'] = new_critic_lr

    def act(self, obs: np.ndarray, explore: bool):
        """Select an action for a given observation."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs_tensor).squeeze(0)
        if explore:
            noise_sample = self.noise.sample().to(self.device)
            action = action + noise_sample
        # Clip action to valid bounds
        action = torch.clamp(action, self.lower_action_bound, self.upper_action_bound)
        return action.cpu().numpy()

    def update(self, batch: Transition) -> dict:
        """Perform one update step for both critic and actor networks."""
        # Unpack batch and ensure tensors are on the proper device
        states, actions, next_states, rewards, dones = batch
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # --- Update Critic ---
        with torch.no_grad():
            # Get next actions from target actor
            next_actions = self.actor_target(next_states)
            # Compute target Q values; detach so gradients do not flow through target network
            target_q_values = self.critic_target(torch.cat([next_states, next_actions], dim=-1)).detach()
            target_values = rewards + self.gamma * (1 - dones) * target_q_values
        # Compute current Q estimates
        predicted_q_values = self.critic(torch.cat([states, actions], dim=-1))
        q_loss = F.mse_loss(predicted_q_values, target_values)

        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        # --- Update Actor ---
        predicted_actions = self.actor(states)
        # The policy loss is the negative expected Q value for the actions chosen by the actor.
        policy_loss = -self.critic(torch.cat([states, predicted_actions], dim=-1)).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # --- Soft update target networks ---
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "q_loss": q_loss.item(),
            "p_loss": policy_loss.item(),
        }
