import numpy as np
import gymnasium as gym
from panda_mujoco_gym.envs.slide import FrankaSlideEnv
from stable_baselines3 import SAC

# reload environment and model
env = gym.make("FrankaSlideSparse-v0", render_mode="human")
model = SAC.load('examples/slide/sac/sac_her_slide_1M', env=env)


# Evaluate the agent
n_eval_episodes = 50
successes = []
episode_returns = []
current_return = 0.0
obs, info = env.reset()

for episode in range(n_eval_episodes):
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        current_return += reward
        done = terminated or truncated

        if done:
            success = info.get("is_success", False)
            successes.append(success)
            episode_returns.append(current_return)
            print(f"[Episode {episode + 1}] Return: {current_return:.2f} | Success: {success}")
            current_return = 0.0
            obs, info = env.reset()

# Summary
print("\n=== Evaluation Summary ===")
print(f"Success Rate: {np.mean(successes):.2%} over {len(successes)} episodes")
print(f"Average Return: {np.mean(episode_returns):.2f}")
print(f"Return Std Dev: {np.std(episode_returns):.2f}")

