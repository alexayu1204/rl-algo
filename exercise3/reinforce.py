import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
from typing import List, Dict, Tuple

class Reinforce:
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        learning_rate: float,
        hidden_size: List[int],
        gamma: float = 0.99,
        **kwargs
    ):
        """
        Initialize the REINFORCE agent.
        :param observation_space: The observation space
        :param action_space: The action space
        :param learning_rate: The learning rate
        :param hidden_size: The sizes of the hidden layers
        :param gamma: The discount factor
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the policy network
        self.policy = nn.Sequential(
            nn.Linear(np.array(observation_space.shape).prod(), hidden_size[0]),
            nn.LayerNorm(hidden_size[0]),
            nn.Tanh(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.LayerNorm(hidden_size[1]),
            nn.Tanh(),
            nn.Linear(hidden_size[1], action_space.n)
        )
        
        # Initialize weights with orthogonal initialization
        for layer in self.policy:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.policy.to(self.device)

    def act(self, observation: np.ndarray, explore: bool = True) -> Tuple[int, torch.Tensor, None]:
        """
        Sample an action from the policy.
        :param observation: The observation
        :param explore: Whether to sample from the policy or take the most likely action
        :return: The action, log probability, and value (None for REINFORCE)
        """
        with torch.no_grad():
            observation = torch.FloatTensor(observation).to(self.device)
            logits = self.policy(observation)
            
            if explore:
                action_distribution = Categorical(logits=logits)
                action = action_distribution.sample()
                log_prob = action_distribution.log_prob(action)
                return action.item(), log_prob, None
            else:
                action = torch.argmax(logits)
                # For evaluation, we don't need the log probability
                return int(action.item()), torch.tensor(0.0).to(self.device), None

    def update(self, batch_info: Dict) -> Dict:
        """
        Update the policy network using the collected batch of data.
        :param batch_info: Dictionary containing batch information
        :return: Dictionary with training metrics
        """
        states = torch.FloatTensor(np.array(batch_info['states'])).to(self.device)
        actions = torch.LongTensor(batch_info['actions']).to(self.device)
        returns = torch.FloatTensor(batch_info['returns']).to(self.device)
        old_log_probs = torch.stack(batch_info['log_probs']).to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Get current action probabilities
        logits = self.policy(states)
        action_distribution = Categorical(logits=logits)
        log_probs = action_distribution.log_prob(actions)
        
        # Calculate ratio and clip
        ratio = torch.exp(log_probs - old_log_probs)
        clip_ratio = 0.2
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        
        # Calculate losses
        policy_loss = -(torch.min(ratio * returns, clipped_ratio * returns)).mean()
        entropy_loss = -0.01 * action_distribution.entropy().mean()
        
        # Total loss
        loss = policy_loss + entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': loss.item(),
            'mean_ratio': ratio.mean().item(),
            'mean_return': returns.mean().item()
        }

    def save(self, path: str):
        """Save the policy network to a file."""
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        """Load the policy network from a file."""
        self.policy.load_state_dict(torch.load(path)) 