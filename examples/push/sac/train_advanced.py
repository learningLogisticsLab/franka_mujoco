# Updated and advanced train.py that includes logging, vectorized environments, and periodic recorded evaluations
import os
import numpy as np
from datetime import datetime

import gymnasium as gym

from panda_mujoco_gym.envs.push import FrankaPushEnv
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics

from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure

# ----------  CONFIG  ----------
ENV_ID = "FrankaPushSparse-v0"

SEED = 42
TOTAL_TIMESTEPS = 500_000
MAX_EPISODE_STEPS = 75
DATETIME = datetime.now()

# Vectorization
N_ENVS = 10 # number of parallel environments for training

# Eval
DESIRED_EVAL_FREQ = 25_000
EVAL_FREQ = DESIRED_EVAL_FREQ // N_ENVS
N_EVAL_EPISODES = 15

# Logs
LOG_DIR = f"/home/student/data/franka_baselines/push/SAC/franka_push_sac_{DATETIME.strftime('%Y-%m-%d_%H:%M:%S')}"
VIDEO_FOLDER = os.path.join(LOG_DIR, "videos")
BEST_MODEL_PATH = os.path.join(LOG_DIR, "best_model")

# --------------------------------

# --------  HELPERS  ------------
# === Vectorized Training Environment ===
def make_env(rank: int, seed: int = 0, render: bool = False):
    """
    Returns a thunk for vec env creation.
    Each env is:
      - Time-limited     (needed for HER relabelling)
      - RecordEpisodeStatistics (keeps ep-return/length per env)
      - Monitor          (with handle_timeout_termination=True)
    """
    def _init():
        env = gym.make(
            ENV_ID,
            render_mode="rgb_array" if render else None,
        )

        # ---- Per-environment wrappers --- #
        # In vec envs -> critical to guarantee per-env episode boundaries for correct relabeling in HER.
        env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)

        # Collect per-env episode statistics  env = RecordEpisodeStatistics(env)

        # Adavned eval stat        env = Monitor(env)

        # Different seed per each worker
        env.reset(seed=seed + rank)
        
        return env
    
    set_random_seed(seed)

    return _init


def main():

    # Create log environments
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(VIDEO_FOLDER, exist_ok=True)
    
    # Initializes parallel training environments with different seeds
    vec_train_env = SubprocVecEnv([make_env(i, seed=42) for i in range(N_ENVS)])

    # Only 1 env for video recording
    eval_env = DummyVecEnv([make_env(rank=0, seed=SEED + 10, render=True)])
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
    n_actions = vec_train_env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))

    # === Define Model ===
    model = SAC(
        policy="MultiInputPolicy",
        env=vec_train_env,
        replay_buffer_class = HerReplayBuffer,
        replay_buffer_kwargs = dict(
            n_sampled_goal = 8, 
            goal_selection_strategy = GoalSelectionStrategy.FUTURE,

            # Store full `info` dict w e/ transition.
            # HER only needs`"is_success" flag (and the dict in your case is empty beyond that). | **Keep False**. Comes at cost of mem without improved.
            copy_info_dict = False,
        ),

        # training hyper-params
        learning_starts=MAX_EPISODE_STEPS*N_ENVS,     # ← wait until at least one episode is in the buffer: max_steps*num_envs
        batch_size=1024,
        train_freq=(1, "step"),
        gradient_steps=N_ENVS,                            # ← keeps updates decorrelated in vec setting
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
    vec_train_env.close()
    eval_env.close()    

if __name__ == "__main__":
    main()
