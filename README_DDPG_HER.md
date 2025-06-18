# DDPG with HER for Franka Pick-and-Place

This directory contains three different implementations of Deep Deterministic Policy Gradient (DDPG) with Hindsight Experience Replay (HER) for the Franka pick-and-place environment.

## Files Overview

### 1. `test_ddpg_her.py` - Basic DDPG Implementation
- Standard DDPG implementation with action noise for exploration
- Uses Stable Baselines3's built-in DDPG
- Good starting point for understanding DDPG

### 2. `test_ddpg_her_enhanced.py` - Custom HER Implementation
- Implements a custom HER buffer that integrates with DDPG
- More complex but potentially more effective
- Includes detailed HER sampling strategies

### 3. `test_ddpg_her_simple.py` - Simple HER Wrapper
- Uses a simple environment wrapper for HER
- Easier to understand and debug
- Recommended for most users

## Key Differences from PPO

### DDPG Advantages for Robotic Tasks:
1. **Continuous Action Spaces**: DDPG is designed for continuous action spaces, making it ideal for robotic manipulation
2. **Deterministic Policy**: Outputs deterministic actions, which can be more stable for precise movements
3. **Off-policy Learning**: Can learn from past experiences stored in replay buffer
4. **Better Exploration**: Uses action noise for exploration instead of stochastic policies

### HER Advantages for Sparse Rewards:
1. **Goal Relabeling**: Replaces original goals with achieved goals from the same episode
2. **Improved Learning**: Helps the agent learn from failed attempts by treating them as successes for different goals
3. **Sparse Reward Handling**: Particularly effective for environments with sparse rewards like pick-and-place

## Usage

### Prerequisites
```bash
pip install stable-baselines3 gymnasium numpy
```

### Running the Code

1. **Basic DDPG** (recommended for beginners):
```bash
python test_ddpg_her.py
```

2. **Enhanced HER** (for advanced users):
```bash
python test_ddpg_her_enhanced.py
```

3. **Simple HER** (recommended for most users):
```bash
python test_ddpg_her_simple.py
```

## Key Parameters

### DDPG Parameters:
- `learning_rate=1e-3`: Higher than PPO for faster learning
- `buffer_size=1000000`: Large replay buffer for experience storage
- `batch_size=256`: Larger batches for stable learning
- `tau=0.005`: Target network update rate
- `gamma=0.98`: Discount factor (slightly lower for sparse rewards)

### HER Parameters:
- `her_ratio=0.8`: 80% of samples use HER goal relabeling
- `goal_strategy='future'`: Sample goals from future timesteps in the same episode

### Action Noise:
- `NormalActionNoise`: Simple Gaussian noise
- `OrnsteinUhlenbeckActionNoise`: Correlated noise (often better for DDPG)

## Expected Performance

With HER, you should see:
1. **Faster Learning**: The agent learns to reach goals more quickly
2. **Higher Success Rate**: Better performance on sparse reward tasks
3. **More Stable Training**: Reduced variance in learning curves

## Monitoring Training

The code includes TensorBoard logging. To monitor training:
```bash
tensorboard --logdir ./franka_ddpg_her_tensorboard/
```

## Tips for Better Performance

1. **Adjust HER Ratio**: Try different values (0.6-0.9) for `her_ratio`
2. **Experiment with Noise**: Try Ornstein-Uhlenbeck noise instead of normal noise
3. **Network Architecture**: Modify the policy and Q-function architectures
4. **Learning Rate**: DDPG can be sensitive to learning rate - try different values
5. **Training Duration**: HER may require more training steps to see benefits

## Troubleshooting

### Common Issues:
1. **Slow Learning**: Try increasing learning rate or reducing batch size
2. **Unstable Training**: Reduce learning rate or increase batch size
3. **Poor Exploration**: Adjust action noise parameters
4. **Memory Issues**: Reduce buffer size if running out of memory

### Environment-Specific Adjustments:
- Modify the goal replacement logic in HER if your environment has different observation structure
- Adjust the reward computation threshold based on your environment's success criteria
- Change the action space normalization if needed

## References

- [DDPG Paper](https://arxiv.org/abs/1509.02971)
- [HER Paper](https://arxiv.org/abs/1707.01495)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/) 