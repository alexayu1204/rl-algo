import gym
import torch
import numpy as np
from exercise3.agents import Reinforce

# Create environment and agent
env = gym.make('Acrobot-v1')
agent = Reinforce(env.action_space, env.observation_space, learning_rate=2e-3, hidden_size=[128, 64], gamma=0.99)

# Load best model
agent.restore('best_Acrobot-v1_reinforce_model.pt')

# Run test episodes
n_episodes = 10
returns = []

for ep in range(n_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.act(obs, explore=False)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    
    returns.append(total_reward)
    print(f'Episode {ep+1}: Return = {total_reward:.2f}')

print(f'\nTest Results:')
print(f'Mean Return: {np.mean(returns):.2f}')
print(f'Std Return: {np.std(returns):.2f}') 