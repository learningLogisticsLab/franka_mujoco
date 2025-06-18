#!/usr/bin/env python3
"""
Simple visualization test for FrankaPickAndPlaceEnv
This script creates a visual demonstration of the environment rendering.
"""

import numpy as np
import time
from panda_mujoco_gym.envs import FrankaPickAndPlaceEnv


def visualize_environment():
    """Create a simple visualization of the environment"""
    print("Starting Franka Pick and Place Environment Visualization")
    print("=" * 50)
    
    # Create environment with human rendering
    env = FrankaPickAndPlaceEnv(reward_type="sparse", render_mode="human")
    
    try:
        # Reset environment
        print("Resetting environment...")
        obs, info = env.reset()
        print(f"Environment reset successful!")
        print(f"Observation keys: {list(obs.keys()) if isinstance(obs, dict) else 'Not a dict'}")
        
        # Display environment info
        print(f"\nEnvironment Information:")
        print(f"- Action space: {env.action_space}")
        print(f"- Observation space: {env.observation_space}")
        print(f"- Render modes: {env.metadata.get('render_modes', 'Not found')}")
        
        # Run a simple demonstration
        print(f"\nRunning demonstration (10 steps)...")
        total_reward = 0
        
        for step in range(10):
            # Take a random action
            action = env.action_space.sample()
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Step {step + 1}: Action = {action}, Reward = {reward:.4f}, Total = {total_reward:.4f}")
            
            # Check if episode ended
            if terminated:
                print(f"Episode terminated at step {step + 1}")
                break
            elif truncated:
                print(f"Episode truncated at step {step + 1}")
                break
            
            # Small delay to see the rendering
            time.sleep(0.2)
        
        print(f"\nDemonstration completed!")
        print(f"Final total reward: {total_reward:.4f}")
        
        # Keep the window open for a moment
        print("Keeping window open for 3 seconds...")
        time.sleep(3)
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        env.close()
        print("Environment closed.")


def test_different_cameras():
    """Test different camera views"""
    print("\n" + "=" * 50)
    print("Testing Different Camera Views")
    print("=" * 50)
    
    # Create environment with rgb_array rendering for camera testing
    env = FrankaPickAndPlaceEnv(reward_type="sparse", render_mode="rgb_array")
    
    try:
        obs, info = env.reset()
        
        # Test different cameras
        cameras = ["watching", "top_down", "side_view", "front_view", "wrist_camera"]
        
        for camera in cameras:
            try:
                print(f"Testing camera: {camera}")
                env.camera_id = camera
                
                # Render with this camera
                render_output = env.render()
                
                if render_output is not None:
                    print(f"  ✓ {camera}: Success (shape: {render_output.shape})")
                else:
                    print(f"  ✗ {camera}: Failed (returned None)")
                    
            except Exception as e:
                print(f"  ✗ {camera}: Error - {e}")
        
    except Exception as e:
        print(f"Error during camera testing: {e}")
    
    finally:
        env.close()


def main():
    """Main function to run visualization tests"""
    print("Franka Pick and Place Environment Visualization Tests")
    print("=" * 60)
    
    # Run the main visualization
    visualize_environment()
    
    # Test different cameras
    test_different_cameras()
    
    print("\n" + "=" * 60)
    print("Visualization tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main() 