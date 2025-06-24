# Updated and advanced train.py that includes logging, vectorized environments, and periodic recorded evaluations
import sys

# sys.path.insert(1, "../../../panda_mujoco_gym")  # Adjust path to include the panda_mujoco_gym package

import os
import random
import numpy as np
import gymnasium as gym
from datetime import datetime

from panda_mujoco_gym.envs.push import FrankaPushEnv
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.logger import configure

# ----------  CONFIG  ----------
ENV_ID = "FrankaPushSparse-v0"

SEED = random.randint(1, 1000)
TOTAL_TIMESTEPS = 500_000
MAX_EPISODE_STEPS = 75
DATETIME = datetime.now()

# Eval
EVAL_FREQ = 5_000
N_EVAL_EPISODES = 15

# Logs
LOG_DIR = f"/home/student/data/franka_baselines/push/SAC/franka_push_sac_single{DATETIME.strftime('%Y-%m-%d_%H:%M:%S')}"
VIDEO_FOLDER = os.path.join(LOG_DIR, "videos")
BEST_MODEL_PATH = os.path.join(LOG_DIR, "best_model")

# ------------- Create Environment -------------
train_env = gym.make(ENV_ID, render_mode='rgb_array')
train_env = TimeLimit(train_env, max_episode_steps=MAX_EPISODE_STEPS)
train_env = Monitor(train_env)

eval_env = gym.make(ENV_ID, render_mode='rgb_array')
eval_env = TimeLimit(eval_env, max_episode_steps=MAX_EPISODE_STEPS)
eval_env = Monitor(eval_env)


# Create log environments
os.makedirs(LOG_DIR, exist_ok=True)
#os.makedirs(VIDEO_FOLDER, exist_ok=True)

#eval_env = VecVideoRecorder(
#    eval_env,
#    VIDEO_FOLDER,
#    record_video_trigger=lambda step: step % EVAL_FREQ == 0,
#    video_length=MAX_EPISODE_STEPS,
#    name_prefix="eval-video"
#)

# === Logger ===
new_logger = configure(LOG_DIR, ["stdout", "tensorboard"])

# === Action Noise ===
# n_actions = vec_train_env.action_space.shape[0]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))

# === Define Model ===
model = SAC(
    policy="MultiInputPolicy",
    env=train_env,
    replay_buffer_class = HerReplayBuffer,
    replay_buffer_kwargs = dict(
        n_sampled_goal = 4, 
        goal_selection_strategy = GoalSelectionStrategy.FUTURE,

        # Store full `info` dict w e/ transition.
        # HER only needs`"is_success" flag (and the dict in your case is empty beyond that). | **Keep False**. Comes at cost of mem without improved.
        copy_info_dict = True,
    ),

    # training hyper-params
    learning_starts=MAX_EPISODE_STEPS,     # ← wait until at least one episode is in the buffer: max_steps*num_envs
    batch_size=256,
    train_freq=(1, "step"),
    gradient_steps=1,                            # ← keeps updates decorrelated in vec setting
    gamma = 0.98,
    learning_rate=1e-3,
    #action_noise=action_noise,
    verbose=1,
    seed=SEED,
    tensorboard_log=LOG_DIR,
)
model.set_logger(new_logger)

# === Evaluation Callback ===
callback_on_best = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=5,
    min_evals=3,
    verbose=1,
)

eval_callback = EvalCallback(
    eval_env,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    render=False,
    best_model_save_path=BEST_MODEL_PATH,
    callback_on_new_best=callback_on_best,
    log_path=LOG_DIR,                              
    verbose=1,
)

# === Train ===
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=True)

# === Save Final Model ===
model.save(os.path.join(LOG_DIR, "final_model"))

# ----------  CLEAN-UP ----------
train_env.close()
eval_env.close()
