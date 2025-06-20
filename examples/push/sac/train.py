import gymnasium as gym
from panda_mujoco_gym.envs.push import FrankaPushEnv
from stable_baselines3 import HerReplayBuffer, SAC

env = gym.make("FrankaPushSparse-v0")

# SAC hyperparams:
model = SAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
    ),
    tensorboard_log="examples/push/sac/logs/sac_her_push",
    verbose=1,
    buffer_size=int(1e6),
    learning_rate=1e-3,
    gamma=0.95,
    batch_size=256,
    policy_kwargs=dict(net_arch=[256, 256, 256]),
)

# set step count and learn. Then save model.
step_count = 2_000
model.learn(step_count)
model.save(f"examples/slide/sac/sac_her_push_{int(step_count/1000)}k")