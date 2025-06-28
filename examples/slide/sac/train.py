import gymnasium as gym
from panda_mujoco_gym.envs.slide import FrankaSlideEnv
from stable_baselines3 import HerReplayBuffer, SAC
from datetime import datetime

TIMESTEPS = 1_000_000
DATETIME = datetime.now()
LOG_DIR = f"/home/student/data/franka_baselines/slide/SAC/franka_push_slide_no_eval{DATETIME.strftime('%Y-%m-%d_%H:%M:%S')}"
env = gym.make("FrankaSlideSparse-v0")

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

# Learn and save model
model.learn(TIMESTEPS)
model.save("/home/student/data/franka_baselines/slide/SAC/models/franka_push_slide_no_eval{DATETIME.strftime('%Y-%m-%d_%H:%M:%S')")
