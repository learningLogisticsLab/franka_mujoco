import gymnasium as gym
import panda_mujoco_gym  # Import to register environments
from stable_baselines3 import DDPG
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

env = gym.make("FrankaPushSparse-v0")

# DDPG hyperparams:
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

model = DDPG(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
    ),
    tensorboard_log="examples/push/ddpg/logs/ddpg_her_push",
    verbose=1,
    buffer_size=int(1e6),
    learning_rate=1e-3,
    gamma=0.99,
    batch_size=256,
    tau=0.005,  # Soft update coefficient for target networks
    action_noise=action_noise,
    policy_kwargs=dict(net_arch=[256, 256, 256]),
)

# set step count and learn. Then save model.
step_count = 1_000_000
model.learn(step_count)
model.save(f"examples/push/ddpg/ddpg_her_push_{int(step_count/1000)}k")