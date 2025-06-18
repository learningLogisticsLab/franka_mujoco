# FR3-with-SB3

This repository integrates the FR3 environment with Stable Baselines 3.

## Components

- [FR3 Environment](https://github.com/CleiverRuiz-LU/FR3_env): Provides the simulation environment for the FR3 robot
- [SB3 Source Code](https://github.com/CleiverRuiz-LU/SB3-SourceCode): Contains the Stable Baselines 3 code used for reinforcement learning

## Setup


cd FR3_env 

pip install -r requirements.txt

pip install -e . 

pip install . 

sudo cp -r /home/student/code/cleiver/franka_mujoco/FR3_env/panda_mujoco_gym/assets /home/student/anaconda3/envs/FR3-SB3_env/lib/python3.10/site-packages/panda_mujoco_gym/