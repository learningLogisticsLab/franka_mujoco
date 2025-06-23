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

import gymnasium as gym
import numpy as np

# Global variables to store episode data across all iterations
actions = []        # List storing action sequences for each episode
observations = []   # List storing observation sequences for each episode  
infos = []         # List storing info dictionaries for each episode


Kp = 6.0            # Proportional control gain for action scaling
robot = 'franka'  # Robot type used in the environment, can be 'franka' or 'fetch'

def main():
    """
    Orchestrates the data generation process by running multiple episodes 
    of the pick-and-place task.
    
    Creates environment, runs scripted episodes, and saves demonstration data
    to compressed NPZ file for use with stable-baselines3.
    """
    # Initialize Fetch pick-and-place environment
    env = gym.make('FrankaPickAndPlaceEnvsparse-v0')
    
    # Configuration parameters
    numItr = 100                    # Number of demonstration episodes to generate
    initStateSpace = "random"       # Initial state space configuration
    
    # Reset environment to initial state
    obs, _ = env.reset()
    print("Reset!")
    
    # Generate demonstration episodes
    while len(actions) < numItr:
        obs,_ = env.reset()
        print("ITERATION NUMBER ", len(actions))
        goToGoal(env, obs)
    
    # Create output filename with configuration details
    fileName = "data_" + robot
    fileName += "_" + initStateSpace
    fileName += "_" + str(numItr)
    fileName += ".npz"
    
    # Save collected data to compressed NPZ file
    np.savez_compressed(fileName, acs=actions, obs=observations, info=infos)
    print(f"Data saved to {fileName}")

def goToGoal(env, lastObs):
    """
    Executes a scripted pick-and-place sequence using a hierarchical approach.
    
    Implements 4-phase control strategy:
    1. Approach: Move gripper above object (3cm offset)
    2. Grasp: Move to object and close gripper  
    3. Transport: Move grasped object to goal position
    4. Maintain: Hold position until episode ends
    
    Args:
        env: Gymnasium environment instance
        lastObs: Last observation containing goal and object state information
            - desired_goal: Target position for object placement
            - observation[3:6]: Current object position (x, y, z)
            - observation[6:9]: Relative position between gripper and object
    """
   # Extract desired position from obs dict
    desired_position = lastObs["desired_goal"][0:3]                             # Target goal position

    # Extract current position from obs dict
    current_position = lastObs["observation"][0:3]
    
    # Relative position to object
    obj_rel_pos = desired_position - current_position  
    
    # Initialize episode data collection
    episodeAcs = []     # Actions for this episode
    episodeObs = []     # Observations for this episode  
    episodeInfo = []    # Info for this episode
    
    # Phase 1: Approach Object (Above)
    # Create target position 3cm above the object. Use copy() method.
    error = object_rel_pos.copy()
    error[2] += 0.03  # Move 3cm above object for safe approach
    
    timeStep = 0  # Track total timesteps in episode
    episodeObs.append(lastObs)
    
    # Phase 1: Move gripper to position above object
    # Terminate when distance to above-object position < 5mm
    while np.linalg.norm(error) >= 0.005 and timeStep <= env._max_episode_steps:
        env.render()  # Visual feedback
        
        # Initialize action vector [x, y, z, gripper]
        action = [0., 0., 0., 0.]
        
        # Update target position (3cm above object)
        error = object_rel_pos.copy()
        error[2] += 0.03
        
        # Proportional control with gain of 6
        # action = Kp * error
        for i in range(len(error)):
            action[i] = error[i] * Kp
        
        # Open gripper for approach
        action[ len(action)-1 ] = 0.05
        
        # Execute action and collect data
        new_obs, reward, done, info = env.step(action)
        timeStep += 1
        
        # Store episode data
        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(new_obs)
        
        # Update state information
        current_position = new_obs["observation"][0:3]
        obj_rel_pos = desired_position - current_position        
    
    # Phase 2: Grasp Object
    # Move gripper directly to object and close gripper
    # Terminate when relative distance to object < 5mm
    while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep <= env._max_episode_steps:
        env.render()
        
        # Initialize action vector [x, y, z, gripper]
        action = [0., 0., 0., 0.]
        
        # Direct proportional control to object position
        for i in range( len(object_rel_pos) ):
            action[i] = object_rel_pos[i] * Kp
        
        # Close gripper to grasp object
        action[len(action)-1] = -0.005  
        
        # Execute action and collect data
        new_obs, reward, done, info = env.step(action)
        timeStep += 1
        
        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(new_obs)
        
        # Update state information
        current_position = new_obs["observation"][0:3]
        obj_rel_pos = desired_position - current_position 
    
    # Phase 3: Transport to Goal
    # Move grasped object to desired goal position
    # Terminate when distance between object and goal < 10mm
    while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= env._max_episode_steps:
        env.render()
        
        action = [0., 0., 0., 0.]
        
        # Proportional control toward goal position
        for i in range(len(goal - objectPos)):
            action[i] = (goal - objectPos)[i] * 6
        
        action[len(action)-1] = -0.005  # Maintain grip on object
        
        # Execute action and collect data
        new_obs, reward, done, info = env.step(action)
        timeStep += 1
        
        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(new_obs)
        
        # Update state information
        objectPos = new_obs['observation'][3:6]
        object_rel_pos = new_obs['observation'][6:9]
    
    # Phase 4: Maintain Position
    # Hold final position until episode completion
    # Continue until maximum episode steps reached
    while True:
        env.render()
        
        # Zero motion command
        action = [0, 0, 0, 0]
        action[len(action)-1] = -0.005  # Keep gripper closed
        
        # Execute action and collect data
        new_obs, reward, done, info = env.step(action)
        timeStep += 1
        
        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(new_obs)
        
        # Update state information
        objectPos = new_obs['observation'][3:6]
        object_rel_pos = new_obs['observation'][6:9]
        
        # Episode termination condition
        if timeStep >= env._max_episode_steps:
            break
    
    # Store complete episode data in global lists
    actions.append(episodeAcs)
    observations.append(episodeObs)
    infos.append(episodeInfo)

if __name__ == "__main__":
    main()
