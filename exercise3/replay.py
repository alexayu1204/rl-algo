"""
Experience replay implementations
"""
from collections import namedtuple
import numpy as np
import torch
from typing import List


Transition = namedtuple(
    "Transition", ("states", "actions", "next_states", "rewards", "done")
)


class ReplayBuffer:
    """Replay buffer to store past experiences that the agent can then use for training data"""

    def __init__(self, size: int, device: torch.device = None):
        """Initialize replay buffer.

        Args:
            size: Maximum number of transitions to store in the buffer
            device: PyTorch device to store the transitions on
        """
        self.size = size
        self.device = device or torch.device("cpu")
        self.reset()

    def reset(self):
        """Reset the buffer to empty."""
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.position = 0
        self.is_full = False

    def add(self, obs: np.ndarray, action, next_obs: np.ndarray, reward: float, done: bool):
        """
        Add a new transition to the buffer.
        For continuous actions (like in DDPG), action is expected to be an array.
        For discrete actions, action is expected to be a scalar.
        """
        # Convert observations to tensors
        state = torch.FloatTensor(obs)
        next_state = torch.FloatTensor(next_obs)
        
        # Check if action is a sequence (e.g. numpy array or list) or a scalar
        if isinstance(action, (np.ndarray, list)):
            # For continuous actions, do not wrap in a list.
            action_tensor = torch.FloatTensor(action)
        else:
            # For discrete actions, wrap it as before.
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


    def sample(self, batch_size: int) -> Transition:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Batch of transitions
        """
        indices = np.random.randint(0, len(self.states), size=min(batch_size, len(self.states)))
        
        states = torch.stack([self.states[idx] for idx in indices]).to(self.device)
        actions = torch.stack([self.actions[idx] for idx in indices]).to(self.device)
        next_states = torch.stack([self.next_states[idx] for idx in indices]).to(self.device)
        rewards = torch.stack([self.rewards[idx] for idx in indices]).to(self.device)
        dones = torch.stack([self.dones[idx] for idx in indices]).to(self.device)

        return Transition(states, actions, next_states, rewards, dones)

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.states)
