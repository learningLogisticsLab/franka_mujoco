import gymnasium as gym
#import sys

#sys.path.append('home/student/code/caeden/franka_mujoco/panda_mujoco_gym')

from panda_mujoco_gym.envs.push import FrankaPushEnv
from stable_baselines3 import HerReplayBuffer, SAC
from datetime import datetime

TIMESTEPS = 500_000
DATETIME = datetime.now()
LOG_DIR = f"/home/bison/code/franka_mujoco/logs/franka_baselines/push/SAC/franka_push_sac_test_{DATETIME.strftime('%Y-%m-%d_%H-%M-%S')}"

env = gym.make("FrankaPushSparse-v0", render_mode='human') # to visualize use render_mode="human"

# SAC hyperparams:
model = SAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
    ),
    tensorboard_log=LOG_DIR,
    verbose=1,
    buffer_size=int(1e6),
    learning_rate=1e-3,
    gamma=0.95,
    batch_size=256,
    policy_kwargs=dict(net_arch=[256, 256, 256]),
)

# set step count and learn. Then save model.
model.learn(TIMESTEPS)
model.save(f"/home/student/data/franka_baselines/push/SAC/models/franka_push_sac_test{DATETIME.strftime('%Y-%m-%d_%H:%M:%S')}")