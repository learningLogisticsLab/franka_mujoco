import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np
from FR3_env.panda_mujoco_gym.envs import FrankaPickAndPlaceEnv

# Create the environment
env = FrankaPickAndPlaceEnv(reward_type="sparse", render_mode="human")
env = DummyVecEnv([lambda: env])

# Define action noise for exploration
# DDPG benefits from action noise for exploration
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
    tensorboard_log="./franka_ddpg_tensorboard/",
    policy_kwargs=dict(
        net_arch=dict(
            pi=[256, 256, 256],  # Policy network architecture
            qf=[256, 256, 256]   # Q-function network architecture
        )
    )
)

# Create evaluation environment for callbacks
eval_env = FrankaPickAndPlaceEnv(reward_type="sparse", render_mode="human")
eval_env = DummyVecEnv([lambda: eval_env])

# Create evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_ddpg_model/",
    log_path="./logs/",
    eval_freq=10000,  # Evaluate every 10k steps
    deterministic=True,
    render=False
)

# Train the model
total_timesteps = 1_000_000  # Adjust this based on your needs
model.learn(
    total_timesteps=total_timesteps,
    callback=eval_callback,
    progress_bar=True
)

# Save the model
model.save("franka_ddpg_model")

# Test the trained model
print("Testing the trained DDPG model...")
obs = env.reset()
total_reward = 0
success_count = 0

for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    total_reward += rewards[0]
    
    # Check for success (assuming the environment provides success info)
    if 'success' in info[0] and info[0]['success']:
        success_count += 1
    
    env.render()
    
    if dones[0]:
        obs = env.reset()
        print(f"Episode {i//100}: Total reward: {total_reward}, Success rate: {success_count/(i+1)*100:.2f}%")

print(f"Final success rate: {success_count/1000*100:.2f}%")
print(f"Average reward per episode: {total_reward/10:.2f}") 