import gymnasium as gym
import numpy as np
import time
import os
import sys

# Import the environment
from panda_mujoco_gym.envs import FrankaPickAndPlaceEnv


def test_environment_creation():
    """Test that the environment can be created successfully"""
    print("Testing environment creation...")
    
    try:
        # Test with human render mode
        env = FrankaPickAndPlaceEnv(reward_type="sparse", render_mode="human")
        print("✓ Environment created successfully with human render mode")
        
        # Test with rgb_array render mode
        env_rgb = FrankaPickAndPlaceEnv(reward_type="sparse", render_mode="rgb_array")
        print("✓ Environment created successfully with rgb_array render mode")
        
        # Test without render mode
        env_no_render = FrankaPickAndPlaceEnv(reward_type="sparse")
        print("✓ Environment created successfully without render mode")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return False


def test_environment_reset():
    """Test that the environment can be reset"""
    print("\nTesting environment reset...")
    
    try:
        env = FrankaPickAndPlaceEnv(reward_type="sparse", render_mode="human")
        
        # Reset the environment
        obs, info = env.reset()
        
        print(f"✓ Environment reset successful")
        print(f"  - Observation keys: {list(obs.keys()) if isinstance(obs, dict) else 'Not a dict'}")
        print(f"  - Info keys: {list(info.keys())}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Failed to reset environment: {e}")
        return False


def test_environment_step():
    """Test that the environment can take steps"""
    print("\nTesting environment step...")
    
    try:
        env = FrankaPickAndPlaceEnv(reward_type="sparse", render_mode="human")
        
        # Reset the environment
        obs, info = env.reset()
        
        # Get action space info
        action_space = env.action_space
        print(f"✓ Action space: {action_space}")
        print(f"  - Shape: {action_space.shape}")
        print(f"  - Low: {action_space.low}")
        print(f"  - High: {action_space.high}")
        
        # Take a random action
        action = action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"✓ Step successful")
        print(f"  - Reward: {reward}")
        print(f"  - Terminated: {terminated}")
        print(f"  - Truncated: {truncated}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Failed to step environment: {e}")
        return False


def test_rendering_modes():
    """Test different rendering modes"""
    print("\nTesting rendering modes...")
    
    try:
        # Test rgb_array render mode
        env = FrankaPickAndPlaceEnv(reward_type="sparse", render_mode="rgb_array")
        obs, info = env.reset()
        
        # Render and check the output
        render_output = env.render()
        
        if render_output is not None:
            print(f"✓ RGB array rendering successful")
            print(f"  - Render output shape: {render_output.shape}")
            print(f"  - Render output dtype: {render_output.dtype}")
        else:
            print("✗ RGB array rendering returned None")
            return False
        
        env.close()
        
        # Test human render mode (just create, don't actually render for long)
        env_human = FrankaPickAndPlaceEnv(reward_type="sparse", render_mode="human")
        obs, info = env_human.reset()
        
        # Just test that render doesn't crash
        env_human.render()
        print("✓ Human render mode test passed")
        
        env_human.close()
        return True
        
    except Exception as e:
        print(f"✗ Failed to test rendering modes: {e}")
        return False


def test_camera_views():
    """Test different camera views"""
    print("\nTesting camera views...")
    
    try:
        env = FrankaPickAndPlaceEnv(reward_type="sparse", render_mode="rgb_array")
        obs, info = env.reset()
        
        # Test different camera configurations
        cameras = ["watching", "top_down", "side_view", "front_view", "wrist_camera"]
        
        for camera in cameras:
            try:
                # Set camera
                env.camera_id = camera
                render_output = env.render()
                
                if render_output is not None:
                    print(f"✓ Camera '{camera}' rendering successful")
                else:
                    print(f"✗ Camera '{camera}' rendering returned None")
                    
            except Exception as e:
                print(f"✗ Camera '{camera}' failed: {e}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Failed to test camera views: {e}")
        return False


def test_environment_properties():
    """Test environment properties and metadata"""
    print("\nTesting environment properties...")
    
    try:
        env = FrankaPickAndPlaceEnv(reward_type="sparse", render_mode="human")
        
        # Check metadata
        print(f"✓ Render modes: {env.metadata.get('render_modes', 'Not found')}")
        print(f"✓ Render FPS: {env.metadata.get('render_fps', 'Not found')}")
        
        # Check action and observation spaces
        print(f"✓ Action space: {env.action_space}")
        print(f"✓ Observation space: {env.observation_space}")
        
        # Check if environment has required methods
        required_methods = ['reset', 'step', 'render', 'close']
        for method in required_methods:
            if hasattr(env, method):
                print(f"✓ Method '{method}' exists")
            else:
                print(f"✗ Method '{method}' missing")
                return False
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Failed to test environment properties: {e}")
        return False


def test_basic_simulation():
    """Test a basic simulation run"""
    print("\nTesting basic simulation...")
    
    try:
        env = FrankaPickAndPlaceEnv(reward_type="sparse", render_mode="human")
        
        # Reset environment
        obs, info = env.reset()
        
        # Run a few steps
        total_reward = 0
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"  Step {step + 1}: Reward = {reward:.4f}, Total = {total_reward:.4f}")
            
            if terminated or truncated:
                print(f"  Episode ended at step {step + 1}")
                break
            
            # Small delay to see the rendering
            time.sleep(0.1)
        
        print(f"✓ Basic simulation completed successfully")
        print(f"  - Total steps: {step + 1}")
        print(f"  - Total reward: {total_reward:.4f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Failed to run basic simulation: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("FRANKA PICK AND PLACE ENVIRONMENT RENDERING TESTS")
    print("=" * 60)
    
    tests = [
        test_environment_creation,
        test_environment_reset,
        test_environment_step,
        test_rendering_modes,
        test_camera_views,
        test_environment_properties,
        test_basic_simulation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The environment renders correctly.")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
