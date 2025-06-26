# Updated and advanced train.py that includes logging, vectorized environments, and periodic recorded evaluations
import os
from pathlib import Path
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
    
    # Load all demo data. Structure: var_name[num_demo][time_step][key if dict] = value
    data = np.load(demo_npz_path, allow_pickle=True)

    obs_buffer       = data['obs']          # length T+1
    act_buffer       = data['acs']          # length T
    rew_buffer       = data['rewards']      # length T
    term_buffer      = data['terminateds']  # length T
    trunc_buffer     = data['truncateds']   # length T
    info_buffer      = data['info']         # length T
    done_buffer      = data['dones']        # length T, if available

    # Extract number of demonstrations
    num_demos = obs_buffer.shape[0]

    # Extract rollout data for a single episode
    for ep in range(num_demos):
        ep_obs   = obs_buffer[ep]    # this is a length‐(T+1) array of dicts
        ep_acts  = act_buffer[ep]    # length‐T array of actions
        ep_rews  = rew_buffer[ep]
        ep_terms = term_buffer[ep]
        ep_trunc = trunc_buffer[ep]
        ep_done  = done_buffer[ep]
        ep_info  = info_buffer[ep]   # length‐T array of dicts

        # Length of episode:
        T = len(ep_acts)

        # Extract single transitions from the episode data
        for t in range(T):
            # raw single‐step data:
            obs_t      = ep_obs[t]       # dict[str, np.ndarray]  (obs_dim,)
            next_obs_t = ep_obs[t+1]
            a_t        = ep_acts[t]      # np.ndarray (action_dim,)
            r_t        = float(ep_rews[t])
            done_t     = bool(ep_done[t] or ep_terms[t] or ep_trunc[t])

            # Rehydrate info dict and inject the timeout flag
            raw_info     = ep_info[t]      # dict[str,Any]
            if isinstance(raw_info, str):
                import ast
                info_t = ast.literal_eval(raw_info)
            else:
                info_t = raw_info.copy()  
            # Append truncated information to info_t
            info_t["TimeLimit.truncated"] = bool(ep_trunc[t])                      

            # **Add the required batch‐dimension** for n_envs=1 (necessary for defualt DummyVecEnv)
            obs_batch      = {k: v[None, ...] for k, v in obs_t.items()}
            next_obs_batch = {k: v[None, ...] for k, v in next_obs_t.items()}
            action_batch   = a_t[None, ...]            # shape (1, action_dim)
            reward_batch   = np.array([r_t])           # shape (1,)
            done_batch     = np.array([done_t])        # shape (1,)
            infos_batch    = [info_t]                  # length‐1 list

            model.replay_buffer.add(
                obs      = obs_batch,
                next_obs = next_obs_batch,
                action   = action_batch,
                reward   = reward_batch,
                done     = done_batch,
                infos    = infos_batch,
                )

    print(f"Loaded {num_demos} transitions into HER buffer "
          f"(combine_done={combine_done}).")

def get_demo_path(relative_path: str) -> str:
    """
    Given a path relative to this script file, return
    the absolute, normalized path as a string.

    Example:
        # If your demos live at ../../../demos/data.npz
        demo_file = get_demo_path("../../../demos/data_franka_random_10.npz")
    """
    # 1) Resolve this script’s directory
    script_dir = Path(__file__).resolve().parent

    # 2) Join with the user-supplied relative path and normalize
    full_path = (script_dir / relative_path).resolve()

    return str(full_path)


def main():
    """
    Main function to run the training script.
    """

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
    log_file = get_demo_path("")
    LOG_DIR = log_file + f"/franka_pick_n_place_sac_single_demo_{DATETIME.strftime('%Y-%m-%d_%H:%M:%S')}"
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
            #online_sampling = True,  # ← use online sampling for HER -- was for old gym not gymnasium
            
            # Store full `info` dict w e/ transition.
            # HER only needs`"is_success" flag (and the dict in your case is empty beyond that). | **Keep False**. Comes at cost of mem without improved.
            copy_info_dict = True,
        ),

        # training hyper-params
        learning_starts=MAX_EPISODE_STEPS,          # ← wait until at least one episode is in the buffer: max_steps*num_envs
        batch_size=256,
        train_freq=(1, "step"),
        gradient_steps=1,                           # ← keeps updates decorrelated in vec setting
        gamma = 0.98,
        learning_rate=1e-3,
        
        #action_noise=action_noise,                 # ←  action noise not optimal for SAC, but can be used with other algorithms like DDPG
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

    # Get abs path to demo file
    demo_file = get_demo_path("../../../demos/data_franka_random_10.npz")

    # Load the demo file into the HER buffer: mutated model.replay_buffer will persist. 
    load_demos_to_her_buffer_gymnasium(model,demo_file, combine_done=True)

    # === Train ===
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=True)

    # === Save Final Model ===
    model.save(os.path.join(LOG_DIR, "final_model"))

    # ----------  CLEAN-UP ----------
    train_env.close()
    eval_env.close()
    print("Training completed and environments closed.")

if __name__ == "__main__":
    main()