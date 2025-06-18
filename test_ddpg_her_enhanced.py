import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np
from panda_mujoco_gym.envs import FrankaPickAndPlaceEnv
from collections import deque
import random

class HERBuffer:
    """
    Hindsight Experience Replay Buffer
    Implements the HER algorithm for sparse reward environments
    """
    def __init__(self, max_size=1000000, her_ratio=0.8, goal_strategy='future'):
        self.max_size = max_size
        self.her_ratio = her_ratio  # Ratio of HER samples vs regular samples
        self.goal_strategy = goal_strategy  # 'future', 'episode', 'random'
        self.buffer = deque(maxlen=max_size)
        self.episode_buffer = []
        
    def add(self, obs, action, reward, next_obs, done, info):
        """Add experience to the buffer"""
        self.episode_buffer.append((obs, action, reward, next_obs, done, info))
        
        if done:
            # Episode finished, add to main buffer with HER
            self._add_episode_with_her()
            self.episode_buffer = []
    
    def _add_episode_with_her(self):
        """Add episode experiences with HER augmentation"""
        episode_length = len(self.episode_buffer)
        
        for i, (obs, action, reward, next_obs, done, info) in enumerate(self.episode_buffer):
            # Add original experience
            self.buffer.append((obs, action, reward, next_obs, done, info))
            
            # Add HER experiences
            if self.her_ratio > 0:
                num_her_samples = int(episode_length * self.her_ratio)
                her_indices = self._sample_her_goals(i, episode_length, num_her_samples)
                
                for her_idx in her_indices:
                    her_obs, her_action, her_reward, her_next_obs, her_done, her_info = self.episode_buffer[her_idx]
                    
                    # Replace goal in current observation with achieved goal from her_idx
                    modified_obs = self._replace_goal(obs, her_info.get('achieved_goal', obs))
                    modified_next_obs = self._replace_goal(next_obs, her_info.get('achieved_goal', next_obs))
                    
                    # Recompute reward based on new goal
                    modified_reward = self._compute_reward(modified_obs, modified_next_obs, her_info.get('achieved_goal', obs))
                    
                    self.buffer.append((modified_obs, action, modified_reward, modified_next_obs, done, info))
    
    def _sample_her_goals(self, current_idx, episode_length, num_samples):
        """Sample goals for HER based on strategy"""
        if self.goal_strategy == 'future':
            # Sample from future timesteps in the same episode
            future_indices = list(range(current_idx + 1, episode_length))
            if future_indices:
                return random.sample(future_indices, min(num_samples, len(future_indices)))
        elif self.goal_strategy == 'episode':
            # Sample from anywhere in the episode
            return random.sample(range(episode_length), min(num_samples, episode_length))
        elif self.goal_strategy == 'random':
            # Sample from random episodes (simplified)
            return random.sample(range(episode_length), min(num_samples, episode_length))
        
        return []
    
    def _replace_goal(self, obs, new_goal):
        """Replace the goal in observation with new goal"""
        # This is a simplified implementation - you may need to adapt based on your env structure
        if isinstance(obs, dict) and 'desired_goal' in obs:
            modified_obs = obs.copy()
            modified_obs['desired_goal'] = new_goal
            return modified_obs
        return obs
    
    def _compute_reward(self, obs, next_obs, achieved_goal):
        """Compute reward based on goal achievement"""
        # Simplified reward computation - adapt based on your environment
        if isinstance(obs, dict) and 'desired_goal' in obs:
            desired_goal = obs['desired_goal']
            distance = np.linalg.norm(achieved_goal - desired_goal)
            return 0.0 if distance < 0.05 else -1.0  # Success threshold
        return 0.0
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        if len(self.buffer) < batch_size:
            return random.sample(list(self.buffer), len(self.buffer))
        return random.sample(list(self.buffer), batch_size)
    
    def __len__(self):
        return len(self.buffer)

class HERDDPG(DDPG):
    """
    DDPG with Hindsight Experience Replay
    """
    def __init__(self, *args, her_ratio=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.her_ratio = her_ratio
        # Replace the replay buffer with HER buffer
        self.replay_buffer = HERBuffer(
            max_size=self.replay_buffer.max_size,
            her_ratio=her_ratio,
            goal_strategy='future'
        )

# Create the environment
env = FrankaPickAndPlaceEnv(reward_type="sparse", render_mode="human")
env = DummyVecEnv([lambda: env])

# Define action noise for exploration
action_noise = NormalActionNoise(
    mean=np.zeros(env.action_space.shape[-1]),
    sigma=0.1 * np.ones(env.action_space.shape[-1])
)

# Alternative: Ornstein-Uhlenbeck noise (often works better for DDPG)
# action_noise = OrnsteinUhlenbeckActionNoise(
#     mean=np.zeros(env.action_space.shape[-1]),
#     sigma=0.1 * np.ones(env.action_space.shape[-1]),
#     theta=0.15,
#     dt=1e-2
# )

# Create the DDPG model with HER
model = HERDDPG(
    "MultiInputPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=1000000,
    learning_starts=1000,
    batch_size=256,
    tau=0.005,
    gamma=0.98,
    train_freq=(1, "step"),
    gradient_steps=1,
    action_noise=action_noise,
    her_ratio=0.8,  # 80% of samples will use HER
    verbose=1,
    tensorboard_log="./franka_ddpg_her_tensorboard/",
    policy_kwargs=dict(
        net_arch=dict(
            pi=[256, 256, 256],  # Policy network architecture
            qf=[256, 256, 256]   # Q-function network architecture
        )
    )
)

# Create evaluation environment
eval_env = FrankaPickAndPlaceEnv(reward_type="sparse", render_mode="human")
eval_env = DummyVecEnv([lambda: eval_env])

# Create evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_ddpg_her_model/",
    log_path="./logs/",
    eval_freq=10000,
    deterministic=True,
    render=False
)

# Train the model
print("Training DDPG with HER...")
total_timesteps = 1_000_000
model.learn(
    total_timesteps=total_timesteps,
    callback=eval_callback,
    progress_bar=True
)

# Save the model
model.save("franka_ddpg_her_model")

# Test the trained model
print("Testing the trained DDPG with HER model...")
obs = env.reset()
total_reward = 0
success_count = 0
episode_count = 0

for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    total_reward += rewards[0]
    
    # Check for success
    if 'success' in info[0] and info[0]['success']:
        success_count += 1
    
    env.render()
    
    if dones[0]:
        episode_count += 1
        obs = env.reset()
        print(f"Episode {episode_count}: Total reward: {total_reward}, Success rate: {success_count/(i+1)*100:.2f}%")

print(f"Final success rate: {success_count/1000*100:.2f}%")
print(f"Average reward per episode: {total_reward/episode_count:.2f}")
print("Training completed with DDPG + HER!") 