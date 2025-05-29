import os
import gymnasium as gym
import panda_mujoco_gym


# Choose your environment ID (check your panda_mujoco_gym/envs/__init__.py)
env_id = "FrankaPickAndPlaceSparse-v0"  # or "PickAndPlace-v0", etc.

os.system(f"python ddpg_continuous_action.py --env-id {env_id} --total-timesteps 1000000")
