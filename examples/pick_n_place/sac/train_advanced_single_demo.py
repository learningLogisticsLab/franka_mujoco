# Updated and advanced train.py that includes logging, vectorized environments, and periodic recorded evaluations
import sys

# sys.path.insert(1, "../../../panda_mujoco_gym")  # Adjust path to include the panda_mujoco_gym package

import os
import random
import numpy as np
import gymnasium as gym
from datetime import datetime

from panda_mujoco_gym.envs.pick_and_place import FrankaPickAndPlaceEnv
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.logger import configure


def load_demos_to_her_buffer_gymnasium(model, demo_npz_path: str, combine_done: bool = True):
    """
    Load a raw Gymnasium-style .npz of expert episodes into model.replay_buffer.

    demo_npz_path must contain at least these arrays:
      - 'episodeObs'         : shape (T+1, *obs_shape*), list of observations
      - 'episodeAcs'         : shape (T, *act_shape*),   list of actions
      - 'episodeRews'        : shape (T,),               list of rewards
      - 'episodeTerminated'  : shape (T,),               list of terminated flags
      - 'episodeTruncated'   : shape (T,),               list of truncated flags
      - 'episodeInfo'        : shape (T,),               list of info dicts

    Parameters
    ----------
    model : SAC
        Your SAC+HER model, already instantiated (and with the correct HerReplayBuffer).
    demo_npz_path : str
        Path to the .npz file you saved from your demo collector.
    combine_done : bool, default=True
        If True, `done = terminated or truncated`.  If False, `done = terminated` only.
    """
    data = np.load(demo_npz_path, allow_pickle=True)
    obs_buffer       = data['episodeObs']        # length T+1
    act_buffer       = data['episodeAcs']        # length T
    rew_buffer       = data['episodeRews']       # length T
    term_buffer      = data['episodeTerminated'] # length T
    trunc_buffer     = data['episodeTruncated']  # length T
    info_buffer      = data['episodeInfo']       # length T

    n_transitions = len(act_buffer)
    for i in range(n_transitions):
        obs_t      = obs_buffer[i]
        next_obs_t = obs_buffer[i + 1]
        action_t   = act_buffer[i]
        reward_t   = rew_buffer[i]
        terminated = bool(term_buffer[i])
        truncated  = bool(trunc_buffer[i])
        done_t     = (terminated or truncated) if combine_done else terminated
        info_t     = info_buffer[i].item() if isinstance(info_buffer[i], np.ndarray) else info_buffer[i]

        # Add into the HER buffer
        model.replay_buffer.add(
            obs=obs_t,
            next_obs=next_obs_t,
            action=action_t,
            reward=reward_t,
            done=done_t,
            info=info_t,
        )

    print(f"Loaded {n_transitions} transitions into HER buffer "
          f"(combine_done={combine_done}).")



# ----------  CONFIG  ----------
ENV_ID = "FrankaPickAndPlaceSparse-v0"

SEED = random.randint(1, 1000)
TOTAL_TIMESTEPS = 1_000_000
MAX_EPISODE_STEPS = 75
DATETIME = datetime.now()

# Eval
EVAL_FREQ = 5_000
N_EVAL_EPISODES = 15

# Logs
LOG_DIR = f"/home/student/data/franka_baselines/pick_n_place/SAC/franka_pick_n_place_sac_single{DATETIME.strftime('%Y-%m-%d_%H:%M:%S')}"
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
        #online_sampling = True,  # ← use online sampling for HER
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

# === Load Demos ===
load_demos_to_her_buffer(model,"")

# === Train ===
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=True)

# === Save Final Model ===
model.save(os.path.join(LOG_DIR, "final_model"))

# ----------  CLEAN-UP ----------
train_env.close()
eval_env.close()
