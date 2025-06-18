import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np
from FR3_env.panda_mujoco_gym.envs import FrankaPickAndPlaceEnv
import random

class SimpleHERWrapper:
    """
    Simple Hindsight Experience Replay wrapper
    This is a simplified version that can be easily understood and modified
    """
    def __init__(self, env, her_ratio=0.8):
        self.env = env
        self.her_ratio = her_ratio
        self.episode_experiences = []
        
    def reset(self):
        self.episode_experiences = []
        return self.env.reset()
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Store experience
        experience = {
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': obs,  # Will be updated in next step
            'done': done,
            'info': info
        }
        
        # Update previous experience's next_obs
        if self.episode_experiences:
            self.episode_experiences[-1]['next_obs'] = obs
            
        self.episode_experiences.append(experience)
        
        # If episode is done, apply HER
        if done:
            self._apply_her()
            
        return obs, reward, done, info
    
    def _apply_her(self):
        """
        Apply Hindsight Experience Replay to the episode
        """
        episode_length = len(self.episode_experiences)
        
        # For each experience in the episode
        for i, experience in enumerate(self.episode_experiences):
            # With probability her_ratio, replace the goal with a future achieved goal
            if random.random() < self.her_ratio and i < episode_length - 1:
                # Sample a future timestep
                future_idx = random.randint(i + 1, episode_length - 1)
                future_experience = self.episode_experiences[future_idx]
                
                # Replace goal with achieved goal from future
                modified_obs = self._replace_goal(experience['obs'], future_experience['info'].get('achieved_goal'))
                modified_next_obs = self._replace_goal(experience['next_obs'], future_experience['info'].get('achieved_goal'))
                
                # Recompute reward
                modified_reward = self._compute_reward(modified_obs, modified_next_obs, future_experience['info'].get('achieved_goal'))
                
                # Update experience
                experience['obs'] = modified_obs
                experience['next_obs'] = modified_next_obs
                experience['reward'] = modified_reward
    
    def _replace_goal(self, obs, new_goal):
        """Replace the goal in observation"""
        if isinstance(obs, dict) and 'desired_goal' in obs:
            modified_obs = obs.copy()
            modified_obs['desired_goal'] = new_goal
            return modified_obs
        return obs
    
    def _compute_reward(self, obs, next_obs, achieved_goal):
        """Compute reward based on goal achievement"""
        if isinstance(obs, dict) and 'desired_goal' in obs:
            desired_goal = obs['desired_goal']
            if achieved_goal is not None:
                distance = np.linalg.norm(achieved_goal - desired_goal)
                return 0.0 if distance < 0.05 else -1.0  # Success threshold
        return 0.0

# Create the environment with HER wrapper
base_env = FrankaPickAndPlaceEnv(reward_type="sparse", render_mode="human")
her_env = SimpleHERWrapper(base_env, her_ratio=0.8)
env = DummyVecEnv([lambda: her_env])

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

# Create the DDPG model
model = DDPG(
    "MultiInputPolicy",
    env,
    learning_rate=1e-3,  # DDPG typically uses higher learning rate than PPO
    buffer_size=1000000,  # Large replay buffer for HER
    learning_starts=1000,  # Start learning after collecting some experiences
    batch_size=256,  # Larger batch size for DDPG
    tau=0.005,  # Target network update rate
    gamma=0.98,  # Slightly lower gamma for sparse rewards
    train_freq=(1, "step"),  # Train every step
    gradient_steps=1,  # Number of gradient steps per update
    action_noise=action_noise,
    verbose=1,
    tensorboard_log="./franka_ddpg_her_simple_tensorboard/",
    policy_kwargs=dict(
        net_arch=dict(
            pi=[256, 256, 256],  # Policy network architecture
            qf=[256, 256, 256]   # Q-function network architecture
        )
    )
)

# Create evaluation environment
eval_base_env = FrankaPickAndPlaceEnv(reward_type="sparse", render_mode="human")
eval_her_env = SimpleHERWrapper(eval_base_env, her_ratio=0.0)  # No HER for evaluation
eval_env = DummyVecEnv([lambda: eval_her_env])

# Create evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_ddpg_her_simple_model/",
    log_path="./logs/",
    eval_freq=10000,  # Evaluate every 10k steps
    deterministic=True,
    render=False
)

# Train the model
print("Training DDPG with Simple HER...")
total_timesteps = 1_000_000  # Adjust this based on your needs
model.learn(
    total_timesteps=total_timesteps,
    callback=eval_callback,
    progress_bar=True
)

# Save the model
model.save("franka_ddpg_her_simple_model")

# Test the trained model
print("Testing the trained DDPG with Simple HER model...")
obs = env.reset()
total_reward = 0
success_count = 0
episode_count = 0

for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    total_reward += rewards[0]
    
    # Check for success (assuming the environment provides success info)
    if 'success' in info[0] and info[0]['success']:
        success_count += 1
    
    env.render()
    
    if dones[0]:
        episode_count += 1
        obs = env.reset()
        print(f"Episode {episode_count}: Total reward: {total_reward}, Success rate: {success_count/(i+1)*100:.2f}%")

print(f"Final success rate: {success_count/1000*100:.2f}%")
print(f"Average reward per episode: {total_reward/episode_count:.2f}")
print("Training completed with DDPG + Simple HER!") 