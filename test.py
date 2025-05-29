import gymnasium as gym
import panda_mujoco_gym

env = gym.make("FrankaPickAndPlaceSparse-v0")
print(env.observation_space)
print(env.observation_space.shape)