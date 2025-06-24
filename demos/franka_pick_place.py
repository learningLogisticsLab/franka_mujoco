"""
Scripted Controller for Franka FR3 Robot - Demonstration Data Generation

This script implements a scripted controller for generating expert demonstration data 
for Deep Reinforcement Learning (DRL) algorithms using the Franka FR3 robot in a 
pick-and-place task. The generated data serves as bootstrapping demonstrations for 
training RL agents with stable-baselines3.

The controller uses a 4-phase hierarchical approach:
1. Approach Object (move gripper above object)
2. Grasp Object (move to object and close gripper)  
3. Transport to Goal (move grasped object to target)
4. Maintain Position (hold final position)

Output: Compressed NPZ file containing action, observation, and info sequences
"""
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from time import sleep
from panda_mujoco_gym.envs import FrankaPickAndPlaceEnv

# Global variables to store episode data across all iterations
observations = []   # List storing observation sequences for each episode  
actions = []        # List storing action sequences for each episode
rewards = []        # List storing reward sequences for each episode
infos = []          # List storing info dictionaries for each episode
terminateds = []    # List storing terminated flags for each episode
truncateds = []     # List storing truncated flags for each episode
dones = []          # List storing done flags (terminated or truncated) for each episode

robot = 'franka'    # Robot type used in the environment, can be 'franka' or 'fetch'

def main():
    """
    Orchestrates the data generation process by running multiple episodes 
    of the pick-and-place task.
    
    Creates environment, runs scripted episodes, and saves demonstration data
    to compressed NPZ file for use with stable-baselines3.
    """
    # Initialize Fetch pick-and-place environment
    env = FrankaPickAndPlaceEnv(reward_type="sparse", render_mode="human")
    env = TimeLimit(env, max_episode_steps=50)  

    # Configuration parameters
    numItr = 50                    # Number of demonstration episodes to generate
    initStateSpace = "random"       # Initial state space configuration
    
    # Reset environment to initial state
    obs, _ = env.reset()
    print("Reset!")
    
    # Generate demonstration episodes
    while len(actions) < numItr:
        obs,_ = env.reset()
        print("ITERATION NUMBER ", len(actions))

        # Execute pick-and-place task
        pick_and_place_demo(env, obs)
    
    # Create output filename with configuration details
    fileName = "data_" + robot
    fileName += "_" + initStateSpace
    fileName += "_" + str(numItr)
    fileName += ".npz"
    
    # Save collected data to compressed numpy NPZ file
    # Set acs,obs,info as keys in dict 
    np.savez_compressed(fileName, 
                        acs = actions, 
                        obs = observations, 
                        rewards = rewards,
                        info = infos)
    
    print(f"Data saved to {fileName}.")

def pick_and_place_demo(env, lastObs):
    """
    Executes a scripted pick-and-place sequence using a hierarchical approach.
    
    Implements 4-phase control strategy:
    1. Approach: Move gripper above object (3cm offset)
    2. Grasp: Move to object and close gripper  
    3. Transport: Move grasped object to goal position
    4. Maintain: Hold position until episode ends

    Store observations, actions, and info in global lists for later replay buffer inclusion.
    
    Args:
        env: Gymnasium environment instance
        lastObs: Last observation containing goal and object state information
            - desired_goal: Target position for object placement
            - observations:
                ee_position[0:3],
                ee_velocity[3:6],
                fingers_width[6],
                object_position[7:10],
                object_rotation[10:13],
                object_velp[13:16],
                object_velr[16:19],
    """

    ## Init goal, current_pos, and object position from last observation
    goal             = np.zeros(3, dtype=np.float32)
    current_pos      = np.zeros(3, dtype=np.float32)
    object_pos       = np.zeros(3, dtype=np.float32)
    object_rel_pos   = np.zeros(3, dtype=np.float32)
    fgr_pos          = np.zeros(1, dtype=np.float32)

    # Initialize episode data collection
    episodeObs  = []        # Observations for this episode  
    episodeAcs  = []        # Actions for this episode
    episodeRews = []        # Rewards for this episode
    episodeInfo = []        # Info for this episode
    episodeTerminated = []  # Terminated flags for this episode
    episodeTruncated = []   # Truncated flags for this episode
    episodeDones = []       # Done flags (terminated or truncated) for this episode

    # Proportional control gain for action scaling -- empirically tuned
    Kp = 8.0            

    # pre_pick_offset
    pre_pick_offset = np.array([0,0,0.03], dtype=float)  # Offset to approach object safely (3cm)
    error_threshold = 0.01  # Threshold for stopping condition (Xmm)
    finger_delta_fast = 0.05   # Action delta for fingers 7.5mm per step. 
    finger_delta_slow = 0.005   # Franka has a range from 0 to 4cm per finger

    ## Extract data
    # Extract desired position from desired_goal dict
    goal = lastObs["desired_goal"][0:3]
    
    # Current robot end-effector position from observation dict
    current_pos = lastObs["observation"][0:3]
    
    # Current object position from observation dict: 
    object_pos = lastObs["observation"][7:10]
    
    # Relative position between end-effector and object
    object_rel_pos = object_pos - current_pos  
    
    ## Phase 1: Approach Object (Above)
    # Create target position 3cm above the object. Use copy() method.
    error = object_rel_pos.copy() 
    error+=pre_pick_offset  # Move 3cm above object for safe approach. Fingers should still end up surrounding object.
    
    timeStep = 0  # Track total timesteps in episode
    episodeObs.append(lastObs)
    
    # Phase 1: Move gripper to position above object
    # Terminate when distance to above-object position < 5mm
    print(f"----------------------------------------------- Phase 1: Approach Object -----------------------------------------------")
    while np.linalg.norm(error) >= error_threshold and timeStep <= env._max_episode_steps:
        env.render()  # Visual feedback
        
        # Initialize action vector [x, y, z, gripper]
        action = np.array([0., 0., 0., 0.])
        
        # Proportional control with gain of 6
        # action = Kp * error
        action[:3] = error * Kp
        
        # Open gripper for approach
        action[ len(action)-1 ] = 0.05
        
        # Unpack new Gymnasium step API
        new_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        timeStep += 1
        
        # Store episode data
        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeRews.append(reward)
        episodeObs.append(new_obs)
        episodeTerminated.append(terminated)
        episodeTruncated.append(truncated)
        episodeDones.append(done)

        # Update state information
        fgr_pos          = new_obs["observation"][6]  
        current_pos = new_obs["observation"][0:3]
        object_pos  = new_obs['observation'][7:10]  
        error            = (object_pos+pre_pick_offset) - current_pos  # Error with regard to offset position         

        # Print debug information
        print(
                f"Time Step: {timeStep}, Error: {np.linalg.norm(error):.4f}, "
                f"Eff_pos: {np.array2string(current_pos, precision=3)}, "
                f"obj_pos: {np.array2string(object_pos, precision=3)}, "
                f"fgr_pos: {np.array2string(fgr_pos, precision=2)}, "
                f"Error: {np.array2string(error, precision=3)}, "
                f"Action: {np.array2string(action, precision=3)}"
                )
        
    # Phase 2: Descend Grasp Object
    # Move gripper directly to object and close gripper
    # Terminate when relative distance to object < 5mm
    print(f"----------------------------------------------- Phase 2: Grip -----------------------------------------------")
    error = object_pos - current_pos # remove offset
    while (np.linalg.norm(error) >= error_threshold or fgr_pos>=0.04) and timeStep <= env._max_episode_steps: # Cube of width 4cm, each finger open to 2cm
        env.render()
        
        # Initialize action vector [x, y, z, gripper]
        action = np.array([0., 0., 0., 0.])
        
        # Direct proportional control to object position
        action[:3] = error * Kp
        
        # Close gripper to grasp object
        action[len(action)-1] = -finger_delta_fast  
        
        # Execute action and collect data
        new_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        timeStep += 1
        
        # Store episode data
        episodeObs.append(new_obs)
        episodeRews.append(reward)
        episodeAcs.append(action)
        episodeInfo.append(info)     
        episodeTerminated.append(terminated)
        episodeTruncated.append(truncated)
        episodeDones.append(done)           

        # Update state information
        fgr_pos = new_obs["observation"][6]  
        current_pos = new_obs["observation"][0:3]
        object_pos = new_obs['observation'][7:10]
        error = object_pos - current_pos 

       # Print debug information
        print(
                f"Time Step: {timeStep}, Error: {np.linalg.norm(error):.4f}, "
                f"Eff_pos: {np.array2string(current_pos, precision=3)}, "
                f"obj_pos: {np.array2string(object_pos, precision=3)}, "
                f"fgr_pos: {np.array2string(fgr_pos, precision=3)}, "
                f"Error: {np.array2string(error, precision=3)}, "
                f"Action: {np.array2string(action, precision=3)}"
                )    
        sleep(0.5)  # Optional: Slow down for better visualization
    
    # Phase 3: Transport to Goal
    # Move grasped object to desired goal position
    # Terminate when distance between object and goal < 1cm
    print(f"----------------------------------------------- Phase 3: Transport to Goal -----------------------------------------------")
    error = goal - current_pos
    while np.linalg.norm(error) >= 0.01 and timeStep <= env._max_episode_steps:
        env.render()
        
        action = np.array([0., 0., 0., 0.])
        
        # Proportional control toward goal position
        action[:3] = error[:3] * Kp
        
        # Maintain grip on object
        #action[len(action)-1] = 0  
        
        # Execute action and collect data
        new_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        timeStep += 1
        
        # Store episode data
        episodeObs.append(new_obs)
        episodeRews.append(reward)
        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeTerminated.append(terminated)
        episodeTruncated.append(truncated)
        episodeDones.append(done)        
        
        # Update state information
        fgr_pos = new_obs["observation"][6] 
        current_pos = new_obs["observation"][0:3]
        object_pos = new_obs['observation'][7:10]
        error = goal - current_pos

       # Print debug information
        print(
                f"Time Step: {timeStep}, Error: {np.linalg.norm(error):.4f}, "
                f"Eff_pos: {np.array2string(current_pos, precision=3)}, "
                f"goal_pos: {np.array2string(goal, precision=3)}, "
                f"fgr_pos: {np.array2string(fgr_pos, precision=2)}, "
                f"Error: {np.array2string(error, precision=3)}, "
                f"Action: {np.array2string(action, precision=3)}"
                )    
    
    # Phase 4: Maintain Position
    # Hold final position until episode completion
    # Continue until maximum episode steps reached
    print(f"----------------------------------------------- Phase 4: Maintain Position -----------------------------------------------")
    while True:
        env.render()
        
        # Zero motion command
        action = np.array([0., 0., 0., 0.])
        #action[len(action)-1] = -0.005  # Keep gripper closed
        action[:3] = error * Kp
        
        # Execute action and collect data
        new_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        timeStep += 1
        
        # Store episode data
        episodeObs.append(new_obs)
        episodeRews.append(reward)
        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeTerminated.append(terminated)
        episodeTruncated.append(truncated)
        episodeDones.append(done)        
        
        # Update state information
        object_pos = new_obs['observation'][7:10]
        current_pos = new_obs["observation"][0:3]
        object_pos = new_obs['observation'][7:10]
        fgr_pos = new_obs["observation"][6]
        error = goal - current_pos

       # Print debug information
        print(
                f"Time Step: {timeStep}, Error: {np.linalg.norm(error):.4f}, "
                f"Eff_pos: {np.array2string(current_pos, precision=3)}, "
                f"obj_pos: {np.array2string(object_pos, precision=3)}, "
                f"goal_pos: {np.array2string(goal, precision=3)}, "
                f"fgr_pos: {np.array2string(fgr_pos, precision=2)}, "
                f"Error: {np.array2string(error, precision=3)}, "
                f"Action: {np.array2string(action, precision=3)}"
                )       

    
        # Episode termination condition
        if timeStep >= env._max_episode_steps:
            break
    
    # Store complete episode data in global lists
    actions.append(episodeAcs)
    observations.append(episodeObs)
    infos.append(episodeInfo)
    rewards.append(episodeRews)

    # Optionally, also store the done/terminated/truncated flags globally if needed:
    terminateds.append(episodeTerminated)
    truncateds.append(episodeTruncated)
    dones.append(episodeDones)    

if __name__ == "__main__":
    main()