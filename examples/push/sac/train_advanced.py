# Updated and advanced train.py that includes logging, vectorized environments, and periodic recorded evaluations
import os
import numpy as np
import gymnasium as gym

from panda_mujoco_gym.envs.push import FrankaPushEnv

from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure

# === Config ===
ENV_ID = "FrankaPushSparse-v0"
TOTAL_TIMESTEPS = 500_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5
MAX_EPISODE_LENGTH = 50
LOG_DIR = "./logs/franka_slide"
VIDEO_FOLDER = os.path.join(LOG_DIR, "videos")
BEST_MODEL_PATH = os.path.join(LOG_DIR, "best_model")
N_ENVS = 4  # number of parallel environments for training

os.makedirs(VIDEO_FOLDER, exist_ok=True)

# === Vectorized Training Environment ===
def make_train_env(rank: int, seed: int = 0):
    def _init():
        env = gym.make(ENV_ID)
        env = Monitor(env)
        env.reset(seed=seed + rank)  # Set different seed per env
        return env
    set_random_seed(seed)
    return _init

# Initializes parallel training environments with different seeds
vec_train_env = SubprocVecEnv([make_train_env(i, base_seed=42) for i in range(N_ENVS)])

# === Single Evaluation Environment (Vec + Video) ===
def make_eval_env():
    def _init():
        env = gym.make(ENV_ID, render_mode="rgb_array")
        env = Monitor(env)
        return env
    return _init

# Only 1 env for video recording
eval_env = DummyVecEnv([make_eval_env()])
eval_env = VecVideoRecorder(
    eval_env,
    VIDEO_FOLDER,
    record_video_trigger=lambda step: step % EVAL_FREQ == 0,
    video_length=MAX_EPISODE_LENGTH,
    name_prefix="eval-video"
)

# === Logger ===
new_logger = configure(LOG_DIR, ["stdout", "tensorboard"])

# === Action Noise ===
n_actions = vec_train_env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# === Define Model ===
model = SAC(
    policy="MultiInputPolicy",
    env=vec_train_env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
        online_sampling=True,
        max_episode_length=MAX_EPISODE_LENGTH,
    ),
    gradient_steps=-1,
    verbose=1,
    seed=0,
    action_noise=action_noise,
    batch_size=256,
    learning_rate=1e-3,
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
    callback_on_new_best=callback_on_best,
    best_model_save_path=BEST_MODEL_PATH,
    log_path=LOG_DIR,
    eval_freq=EVAL_FREQ,
    deterministic=True,
    render=False,
    n_eval_episodes=N_EVAL_EPISODES,
    verbose=1,
)

# === Train ===
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)

# === Save Final Model ===
model.save(os.path.join(LOG_DIR, "final_model"))
