import gymnasium as gym
from panda_mujoco_gym.envs.push import FrankaPushEnv

from stable_baselines3 import SAC
env = gym.make("FrankaPushDense-v0", render_mode="rgb_array")

model = SAC("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=1000, log_interval=4)
model.save("sac_push")

del model

model = SAC.load("sac_push")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
