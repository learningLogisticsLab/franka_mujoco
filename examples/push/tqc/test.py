import gymnasium as gym
from panda_mujoco_gym.envs.push import FrankaPushEnv
from stable_baselines3 import HerReplayBuffer, SAC

# reload environment and model
env = gym.make("FrankaPushSparse-v0", render_mode="human")
model = SAC.load('examples/push/sac/tqc_push', env=env)

# reset
obs, _ = env.reset()

# Evaluate the agent
episode_returns = 0
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = truncated or terminated
    episode_returns += reward
    if done or info.get("is_success", False):
        print("Reward:", episode_returns, "Success?", info.get("is_success", False))
        episode_returns = 0.0
        obs, _ = env.reset()