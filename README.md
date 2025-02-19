# Reinforcement Learning Implementations

This repository contains implementations of various reinforcement learning algorithms, with a focus on the Acrobot-v1 environment. The implementations include DQN and REINFORCE algorithms with several enhancements and optimizations.

## Table of Contents
- [Installation](#installation)
- [Algorithms](#algorithms)
  - [REINFORCE](#reinforce)
  - [DQN](#dqn)
- [Environments](#environments)
- [Performance Improvements](#performance-improvements)
- [Usage](#usage)
- [Results](#results)

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repo-name]
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Algorithms

### REINFORCE
The REINFORCE implementation includes several enhancements:

1. **Network Architecture**:
   - Larger network capacity [256, 128]
   - Layer Normalization for better training stability
   - GELU activation functions
   - Dropout layers (0.1) for regularization
   - Orthogonal initialization with environment-specific gains

2. **Training Dynamics**:
   - Increased learning rate (3e-3)
   - Cosine learning rate decay with warmup
   - Adaptive entropy coefficient
   - Gradient clipping
   - GAE (Generalized Advantage Estimation)

3. **Reward Shaping**:
   - Progressive height-based rewards
   - Momentum-based rewards
   - Action diversity encouragement
   - Consistency bonuses
   - Smart penalties for stillness

### DQN
The DQN implementation features:

1. **Network Architecture**:
   - Double DQN with target network
   - Efficient architecture with Layer Normalization
   - ReLU activation functions
   - Orthogonal initialization

2. **Training Improvements**:
   - Dynamic reward normalization
   - Soft target updates
   - Huber loss for stability
   - Gradient clipping
   - Optimized replay buffer

## Environments

### Acrobot-v1
- **State Space**: 6-dimensional
- **Action Space**: 3 discrete actions
- **Goal**: Swing the end of the double pendulum above the base
- **Success Threshold**: -63 (average reward)

### Other Supported Environments
- CartPole-v1
- LunarLander-v2

## Performance Improvements

### REINFORCE Enhancements
1. **Reward Shaping**:
   ```python
   if height > best_height:
       height_improvement = height - best_height
       reward += 0.3 * height_improvement
       best_height = height
       consecutive_good_actions += 1
   ```

2. **Momentum Rewards**:
   ```python
   if height > -1.5:
       momentum_reward = 0.15 * (abs(angular_vel1) + abs(angular_vel2))
       if height > 0:
           momentum_reward *= 1.5
       reward += momentum_reward
   ```

3. **Action Diversity**:
   ```python
   if last_action == action:
       action_repeat_count += 1
       if action_repeat_count > 2:
           reward -= 0.2 * action_repeat_count
   ```

### Architecture Improvements
```python
self.policy = nn.Sequential(
    nn.Linear(STATE_SIZE, hidden_size[0]),
    nn.LayerNorm(hidden_size[0]),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(hidden_size[0], hidden_size[1]),
    nn.LayerNorm(hidden_size[1]),
    nn.GELU(),
    nn.Linear(hidden_size[1], ACTION_SIZE)
)
```

## Usage

### Training REINFORCE
```bash
# Train REINFORCE on Acrobot-v1
python exercise3/train_reinforce.py --env Acrobot-v1

# Train with specific seed
python exercise3/train_reinforce.py --env Acrobot-v1 --seed 42
```

### Training DQN
```bash
# Train DQN on Acrobot-v1
python exercise3/train_dqn.py --env Acrobot-v1

# Train with rendering disabled
python exercise3/train_dqn.py --env Acrobot-v1 --no-render
```

### Evaluation
```bash
# Evaluate REINFORCE
python exercise3/evaluate_reinforce.py --env Acrobot-v1 --episodes 10

# Evaluate DQN
python exercise3/evaluate_dqn.py --env Acrobot-v1 --episodes 10
```

## Results

### Acrobot-v1 Performance
- **Mean Return**: -111.50
- **Standard Deviation**: 30.08
- **Best Episode**: -82.00
- **Worst Episode**: -194.00
- **Success Rate**: Episodes with return > -120: 80%

### Training Parameters
```python
ENV_CONFIGS = {
    "Acrobot-v1": {
        "max_steps": 500,
        "success_threshold": -63,
        "learning_rate": 3e-3,
        "hidden_size": [256, 128],
        "gamma": 0.995,
        "max_timesteps": 200000,
        "eval_freq": 2000,
        "eval_episodes": 5,
        "max_time": 7200,
    }
}
```

## Contributing
Feel free to submit issues and enhancement requests!

## License
[Your chosen license] 