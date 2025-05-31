import os
import gymnasium as gym
import panda_mujoco_gym
import random

# Set up your experiment settings
env_id = "FrankaPickAndPlaceSparse-v0"
exp_name = "DDPG_with_FR3_env"
total_timesteps = 1_000_000
seed = 1

cmd = (
    f"python3 ddpg_continuous_action.py "
    f"--env-id {env_id} "
    f"--total-timesteps {total_timesteps} "
    f"--seed {seed} "
    f"--cuda "
)



print("Launching experiment with command:")
print(cmd)
os.system(cmd)
