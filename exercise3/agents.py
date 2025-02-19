from abc import ABC, abstractmethod
from copy import deepcopy
import gym
import numpy as np
import os.path
import math
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.categorical import Categorical
import torch.nn
from torch.optim import Adam
from typing import Dict, Iterable, List
import copy
import torch.nn as nn
import torch.optim as optim
import itertools

from exercise3.networks import FCNetwork
from exercise3.replay import Transition


class Agent(ABC):
    """Base class for Deep RL Exercise 3 Agents

    **DO NOT CHANGE THIS CLASS**

    :attr action_space (gym.Space): action space of used environment
    :attr observation_space (gym.Space): observation space of used environment
    :attr saveables (Dict[str, torch.nn.Module]):
        mapping from network names to PyTorch network modules

    Note:
        see http://gym.openai.com/docs/#spaces for more information on Gym spaces
    """

    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        """The constructor of the Agent Class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        """
        self.action_space = action_space
        self.observation_space = observation_space

        self.saveables = {}

    def save(self, path: str, suffix: str = "") -> str:
        """Saves saveable PyTorch models under given path

        The models will be saved in directory found under given path in file "models_{suffix}.pt"
        where suffix is given by the optional parameter (by default empty string "")

        :param path (str): path to directory where to save models
        :param suffix (str, optional): suffix given to models file
        :return (str): path to file of saved models file
        """
        torch.save(self.saveables, path)
        return path

    def restore(self, save_path: str):
        """Restores PyTorch models from models file given by path

        :param save_path (str): path to file containing saved models
        """
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    @abstractmethod
    def act(self, obs: np.ndarray):
        ...

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def update(self):
        ...


class DQN(Agent):
    """The DQN agent for exercise 3

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**

    :attr critics_net (FCNetwork): fully connected DQN to compute Q-value estimates
    :attr critics_target (FCNetwork): fully connected DQN target network
    :attr critics_optim (torch.optim): PyTorch optimiser for DQN critics_net
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr update_counter (int): counter of updates for target network updates
    :attr target_update_freq (int): update frequency (number of iterations after which the target
        networks should be updated)
    :attr batch_size (int): size of sampled batches of experience
    :attr gamma (float): discount rate gamma
    """

    def __init__(self, observation_space, action_space, **kwargs):
        """Initialize the DQN agent with efficient architecture."""
        super().__init__(action_space, observation_space)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Network architecture
        self.critics = nn.Sequential(
            nn.Linear(observation_space.shape[0], kwargs["hidden_size"][0]),
            nn.LayerNorm(kwargs["hidden_size"][0]),
            nn.ReLU(),
            nn.Linear(kwargs["hidden_size"][0], kwargs["hidden_size"][1]),
            nn.LayerNorm(kwargs["hidden_size"][1]),
            nn.ReLU(),
            nn.Linear(kwargs["hidden_size"][1], action_space.n)
        ).to(self.device)
        
        # Initialize weights using orthogonal initialization
        for m in self.critics.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.414)
                nn.init.constant_(m.bias, 0)
        
        self.critics_target = copy.deepcopy(self.critics)
        
        # Optimized Adam parameters
        self.critics_optim = optim.Adam(
            self.critics.parameters(),
            lr=kwargs["learning_rate"],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.batch_size = kwargs["batch_size"]
        self.gamma = kwargs["gamma"]
        self.tau = kwargs["tau"]
        self.critic_clip = 1.0
        
        # Exploration parameters
        self.initial_exploration = 1000
        self.steps = 0
        self.epsilon = kwargs.get("epsilon_start", 1.0)
        self.epsilon_decay = kwargs.get("epsilon_decay", 0.99)
        self.epsilon_min = kwargs.get("epsilon_min", 0.01)
        self.epsilon_decay_strategy = kwargs.get("epsilon_decay_strategy", "exponential")
        
        self.update_counter = 0
        self.target_update_freq = kwargs["target_update_freq"]
        
        # Reward normalization
        self.running_mean = None
        self.running_std = None
        self.reward_alpha = 0.1

        # Add saveables
        self.saveables.update({
            "critics": self.critics,
            "critics_target": self.critics_target,
            "critics_optim": self.critics_optim
        })

    def act(self, obs: np.ndarray, explore: bool):
        """Select an action from the input state."""
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.action_space.n)
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            q_values = self.critics(obs_tensor)
            return q_values.argmax().item()

    def update(self, batch):
        """Update the agent's networks with improved learning."""
        self.steps += 1
        
        # Convert batch data to tensors
        states = torch.FloatTensor(batch.states).to(self.device)
        actions = torch.LongTensor(batch.actions).reshape(-1, 1).to(self.device)
        next_states = torch.FloatTensor(batch.next_states).to(self.device)
        rewards = torch.FloatTensor(batch.rewards).reshape(-1).to(self.device)
        done = torch.FloatTensor(batch.done).reshape(-1).to(self.device)
        
        # Dynamic reward normalization
        if self.running_mean is None:
            self.running_mean = rewards.mean()
            self.running_std = rewards.std() + 1e-8
        else:
            self.running_mean = (1 - self.reward_alpha) * self.running_mean + self.reward_alpha * rewards.mean()
            self.running_std = (1 - self.reward_alpha) * self.running_std + self.reward_alpha * (rewards.std() + 1e-8)
        
        normalized_rewards = torch.clamp((rewards - self.running_mean) / self.running_std, -10, 10)
        
        # Get current Q values
        current_q_values = self.critics(states)
        current_q_values = current_q_values.gather(1, actions).squeeze(1)

        # Compute target Q values using double Q-learning
        with torch.no_grad():
            next_q_values = self.critics(next_states)
            best_actions = next_q_values.max(1, keepdim=True)[1]
            next_q_values_target = self.critics_target(next_states)
            next_q_values = next_q_values_target.gather(1, best_actions).squeeze(1)
            target_q_values = normalized_rewards + self.gamma * next_q_values * (1 - done)
        
        # Use Huber loss for stability
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize with gradient clipping
        self.critics_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics.parameters(), self.critic_clip)
        self.critics_optim.step()
        
        # Soft update target network
        if self.update_counter % self.target_update_freq == 0:
            for target_param, param in zip(self.critics_target.parameters(), self.critics.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Update epsilon
        if self.steps > self.initial_exploration:
            self.schedule_hyperparameters(self.steps, self.target_update_freq)
        
        return {"loss": loss.item()}

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters based on the current timestep"""
        if self.epsilon_decay_strategy == "exponential":
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        elif self.epsilon_decay_strategy == "linear":
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon - (1 - self.epsilon_min) / max_timestep
            )


class Reinforce(Agent):
    """ The Reinforce Agent for Ex 3

    ** YOU NEED TO IMPLEMENT THE FUNCTIONS IN THIS CLASS **

    :attr policy (FCNetwork): fully connected network for policy
    :attr policy_optim (torch.optim): PyTorch optimiser for policy network
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr gamma (float): discount rate gamma
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        gamma: float,
        **kwargs,
    ):
        """
        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param gamma (float): discount rate gamma
        """
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize reward normalization
        self.reward_running_mean = 0
        self.reward_running_std = 1
        
        # Detect environment type for specific adjustments
        self.env_name = kwargs.get('env_name', 'unknown')
        is_acrobot = 'Acrobot' in str(observation_space)
        
        # Initialize policy network with improved architecture
        self.policy = nn.Sequential(
            nn.Linear(STATE_SIZE, hidden_size[0]),
            nn.LayerNorm(hidden_size[0]),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.LayerNorm(hidden_size[1]),
            nn.GELU(),
            nn.Linear(hidden_size[1], ACTION_SIZE)
        ).to(self.device)
        
        # Initialize value network with same architecture
        self.value = nn.Sequential(
            nn.Linear(STATE_SIZE, hidden_size[0]),
            nn.LayerNorm(hidden_size[0]),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.LayerNorm(hidden_size[1]),
            nn.GELU(),
            nn.Linear(hidden_size[1], 1)
        ).to(self.device)
        
        # Initialize weights using orthogonal initialization
        for net in [self.policy, self.value]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    gain = 1.414 if is_acrobot else 1.0
                    nn.init.orthogonal_(m.weight, gain=gain)
                    nn.init.constant_(m.bias, 0.0)

        # Initialize optimizers with improved parameters
        base_lr = learning_rate * (2.0 if is_acrobot else 1.0)
        self.policy_optim = torch.optim.AdamW(
            self.policy.parameters(), 
            lr=base_lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        self.value_optim = torch.optim.AdamW(
            self.value.parameters(),
            lr=base_lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        self.learning_rate = base_lr
        self.initial_learning_rate = base_lr
        self.gamma = gamma
        
        # Improved training parameters
        if is_acrobot:
            self.entropy_coef = 0.2
            self.min_entropy_coef = 0.05
            self.entropy_decay = 0.9995
            self.reward_scale = 0.5
            self.max_grad_norm = 2.0
            self.reward_alpha = 0.2
            self.min_learning_rate = 1e-4
            self.warmup_steps = 1000
            self.lr_decay = 0.9995
            self.normalize_advantages = True
            self.gae_lambda = 0.97
        else:
            self.entropy_coef = 0.05
            self.min_entropy_coef = 0.01
            self.entropy_decay = 0.999
            self.reward_scale = 1.0
            self.max_grad_norm = 0.5
            self.reward_alpha = 0.05
            self.min_learning_rate = 1e-5
            self.warmup_steps = 500
            self.lr_decay = 0.999
            self.normalize_advantages = True
            self.gae_lambda = 0.95

        self.saveables.update({
            "policy": self.policy,
            "value": self.value,
            "policy_optim": self.policy_optim,
            "value_optim": self.value_optim
        })

    def normalize_reward(self, reward: float) -> float:
        """Normalize rewards using running statistics"""
        reward = reward * self.reward_scale
        self.reward_running_mean = (1 - self.reward_alpha) * self.reward_running_mean + self.reward_alpha * reward
        self.reward_running_std = (1 - self.reward_alpha) * self.reward_running_std + \
                                 self.reward_alpha * abs(reward - self.reward_running_mean)
        
        normalized_reward = (reward - self.reward_running_mean) / (self.reward_running_std + 1e-8)
        return np.clip(normalized_reward, -10, 10)

    def act(self, obs: np.ndarray, explore: bool = True) -> int:
        """Select action using the policy network with improved exploration"""
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        
        with torch.no_grad():
            logits = self.policy(obs_tensor)
            if explore:
                # Add noise to logits for exploration
                is_acrobot = 'Acrobot' in str(self.observation_space)
                noise_scale = max(0.3, self.entropy_coef) if is_acrobot else max(0.1, self.entropy_coef)
                noise = torch.randn_like(logits) * noise_scale
                logits = logits + noise
            
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            
        return action.item()

    def update(
        self, rewards: List[float], observations: List[np.ndarray], actions: List[int],
    ) -> Dict[str, float]:
        """Update policy and value networks with improved learning dynamics"""
        # Convert to tensors and move to device
        obs_tensor = torch.FloatTensor(np.array(observations)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        
        # Calculate discounted returns with reward scaling and GAE
        returns = []
        advantages = []
        next_value = 0
        next_advantage = 0
        
        for r in reversed(rewards):
            # Scale and normalize reward
            r = self.normalize_reward(r)
            
            # Calculate return
            current_return = r + self.gamma * next_value
            returns.insert(0, current_return)
            next_value = current_return
            
            # Calculate advantage using GAE
            delta = r + self.gamma * next_value - next_advantage
            advantage = delta + self.gamma * self.gae_lambda * next_advantage
            advantages.insert(0, advantage)
            next_advantage = advantage
        
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize returns and advantages
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        if self.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get value predictions
        values = self.value(obs_tensor).squeeze(-1)
        value_loss = F.smooth_l1_loss(values, returns)
        
        # Update value network
        self.value_optim.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
        self.value_optim.step()
        
        # Get policy predictions
        logits = self.policy(obs_tensor)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        # Calculate policy loss with entropy regularization
        log_probs = dist.log_prob(actions_tensor)
        entropy = dist.entropy().mean()
        
        policy_loss = -(log_probs * advantages.detach()).mean()
        loss = policy_loss - self.entropy_coef * entropy
        
        # Update policy network
        self.policy_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optim.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "mean_advantage": advantages.mean().item(),
            "mean_return": returns.mean().item(),
            "entropy_coef": self.entropy_coef,
            "learning_rate": self.learning_rate
        }

    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):
        """Schedule hyperparameters for better learning dynamics"""
        # Cosine learning rate decay
        progress = min(1.0, timestep / max_timesteps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        self.learning_rate = self.initial_learning_rate * cosine_decay
        self.learning_rate = max(self.min_learning_rate, self.learning_rate)
        
        for optimizer in [self.policy_optim, self.value_optim]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.learning_rate
        
        # Entropy coefficient decay
        self.entropy_coef = max(
            self.min_entropy_coef,
            self.entropy_coef * self.entropy_decay
        )
