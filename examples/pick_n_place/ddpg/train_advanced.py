# Updated and advanced train.py that includes logging, vectorized environments, and periodic recorded evaluations
import sys

# sys.path.insert(1, "../../../panda_mujoco_gym")  # Adjust path to include the panda_mujoco_gym package

import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from panda_mujoco_gym.envs.push import FrankaPushEnv

from stable_baselines3 import DDPG
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure

# === Config ===
ENV_ID = "FrankaPickAndPlaceSparse-v0"
TOTAL_TIMESTEPS = 500_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5
MAX_EPISODE_LENGTH = 500
LOG_DIR = "./logs/franka_pick-n-place_ddpg"
VIDEO_FOLDER = os.path.join(LOG_DIR, "videos")
BEST_MODEL_PATH = os.path.join(LOG_DIR, "best_model")
N_ENVS = 4  # number of parallel environments for training

os.makedirs(VIDEO_FOLDER, exist_ok=True)

def plot_success_rate_vs_timesteps(log_dir, save_path=None, figsize=(12, 8)):
    """
    Plot and save time steps vs success rate from training logs.
    
    Args:
        log_dir (str): Path to the log directory containing tensorboard events
        save_path (str, optional): Path to save the plot. If None, saves to log_dir
        figsize (tuple): Figure size for the plot
    """
    # Find tensorboard event files
    event_files = []
    for file in os.listdir(log_dir):
        if file.startswith('events.out.tfevents'):
            event_files.append(os.path.join(log_dir, file))
    
    if not event_files:
        print("No tensorboard event files found in log directory")
        return
    
    # Use the most recent event file
    event_file = sorted(event_files)[-1]
    
    try:
        # Load tensorboard data
        ea = EventAccumulator(event_file)
        ea.Reload()
        
        # Get all available tags
        tags = ea.Tags()
        print(f"Available tags: {tags}")
        
        # Look for success rate related metrics
        success_metrics = []
        timesteps = []
        
        # Common success rate metric names
        possible_success_tags = [
            'eval/success_rate',
            'eval/ep_success_rate', 
            'eval/mean_success_rate',
            'success_rate',
            'ep_success_rate'
        ]
        
        # Try to find success rate data
        success_tag = None
        for tag in possible_success_tags:
            if tag in tags['scalars']:
                success_tag = tag
                break
        
        if success_tag is None:
            print("No success rate metrics found. Available scalar tags:")
            for tag in tags['scalars']:
                print(f"  - {tag}")
            return
        
        # Extract success rate data
        success_events = ea.Scalars(success_tag)
        for event in success_events:
            timesteps.append(event.step)
            success_metrics.append(event.value)
        
        if not success_metrics:
            print("No success rate data found in the logs")
            return
        
        # Create the plot
        plt.figure(figsize=figsize)
        plt.plot(timesteps, success_metrics, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Training Timesteps', fontsize=12)
        plt.ylabel('Success Rate', fontsize=12)
        plt.title(f'Success Rate vs Training Timesteps\n{ENV_ID}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.0)
        
        # Add some statistics
        if len(success_metrics) > 1:
            final_success = success_metrics[-1]
            max_success = max(success_metrics)
            avg_success = np.mean(success_metrics)
            
            plt.axhline(y=final_success, color='r', linestyle='--', alpha=0.7, 
                       label=f'Final: {final_success:.3f}')
            plt.axhline(y=max_success, color='g', linestyle='--', alpha=0.7, 
                       label=f'Max: {max_success:.3f}')
            plt.axhline(y=avg_success, color='orange', linestyle='--', alpha=0.7, 
                       label=f'Avg: {avg_success:.3f}')
            plt.legend()
        
        # Save the plot
        if save_path is None:
            save_path = os.path.join(log_dir, 'success_rate_vs_timesteps.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Success rate plot saved to: {save_path}")
        
        # Also save as CSV for further analysis
        csv_path = save_path.replace('.png', '.csv')
        df = pd.DataFrame({
            'timesteps': timesteps,
            'success_rate': success_metrics
        })
        df.to_csv(csv_path, index=False)
        print(f"Success rate data saved to: {csv_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error reading tensorboard logs: {e}")
        print("Trying alternative method...")
        
        # Alternative: try to read from monitor logs
        try:
            monitor_files = []
            for file in os.listdir(log_dir):
                if file.endswith('.monitor.csv'):
                    monitor_files.append(os.path.join(log_dir, file))
            
            if monitor_files:
                print(f"Found monitor files: {monitor_files}")
                # You can add monitor file parsing here if needed
            else:
                print("No monitor files found either")
                
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")

# === Vectorized Training Environment ===
def make_train_env(rank: int, seed: int = 0):
    def _init():
        env = gym.make(ENV_ID)
        env = Monitor(env)
        env.reset(seed=seed + rank)  # Set different seed per env
        return env
    set_random_seed(seed)
    return _init

# === Single Evaluation Environment (Vec + Video) ===
def make_eval_env():
    def _init():
        env = gym.make(ENV_ID, render_mode="rgb_array")
        env = Monitor(env)
        return env
    return _init

def main():

    # Initializes parallel training environments with different seeds
    vec_train_env = SubprocVecEnv([make_train_env(i, seed=42) for i in range(N_ENVS)])

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

    # === Action Noise for DDPG ===
    n_actions = vec_train_env.action_space.shape[0]
    # Ornstein-Uhlenbeck noise is typically better for DDPG than normal noise
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), 
        sigma=0.1 * np.ones(n_actions),
        theta=0.15,
        dt=1e-2
    )

    # === Define DDPG Model with HER ===
    model = DDPG(
        policy="MultiInputPolicy",
        env=vec_train_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            goal_selection_strategy=GoalSelectionStrategy.FUTURE,
            copy_info_dict=False,
        ),
        learning_starts=MAX_EPISODE_LENGTH,     # ← wait until at least one episode is in the buffer
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
    
    # === Plot Success Rate ===
    print("\n" + "="*50)
    print("GENERATING SUCCESS RATE PLOT")
    print("="*50)
    plot_success_rate_vs_timesteps(LOG_DIR)

if __name__ == "__main__":
    main()