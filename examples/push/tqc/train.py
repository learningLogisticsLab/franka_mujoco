import gymnasium as gym
from panda_mujoco_gym.envs.push import FrankaPushEnv
from stable_baselines3 import HerReplayBuffer
from sb3_contrib import TQC

env = gym.make("FrankaPushSparse-v0", render_mode="human")

policy_kwargs = dict(n_critics=2, n_quantiles=25)

model = TQC(
    "MultiInputPolicy", 
    env, 
    replay_buffer_class=HerReplayBuffer,
    top_quantiles_to_drop_per_net=2, 
    verbose=1, 
    tensorboard_log="./examples/push/tqc/logs",
    policy_kwargs=policy_kwargs
)
model.learn(total_timesteps=250_000, log_interval=4)
model.save("tqc_push")