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

    def add(self, obs: np.ndarray, action: int, next_obs: np.ndarray, reward: float, done: bool):
        """Add a new transition to the buffer.

        Args:
            obs: Current state
            action: Action taken
            next_obs: Next state
            reward: Reward received
            done: Whether the episode ended
        """
        # Convert to tensors with correct types
        state = torch.FloatTensor(obs)
        next_state = torch.FloatTensor(next_obs)
        action = torch.LongTensor([action])  # Changed to LongTensor for discrete actions
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([float(done)])

        if len(self.states) < self.size:
            self.states.append(state)
            self.actions.append(action)
            self.next_states.append(next_state)
            self.rewards.append(reward)
            self.dones.append(done)
        else:
            self.states[self.position] = state
            self.actions[self.position] = action
            self.next_states[self.position] = next_state
            self.rewards[self.position] = reward
            self.dones[self.position] = done
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
