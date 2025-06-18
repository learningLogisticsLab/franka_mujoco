# Franka Pick and Place Environment Tests

This directory contains comprehensive tests for the `FrankaPickAndPlaceEnv` to verify that it renders correctly and functions properly.

## Test Files

### 1. `test.py` - Comprehensive Test Suite
A complete test suite that verifies all aspects of the environment:

- **Environment Creation**: Tests creating the environment with different render modes
- **Environment Reset**: Verifies the environment can be reset properly
- **Environment Step**: Tests taking actions and receiving observations
- **Rendering Modes**: Tests both `human` and `rgb_array` render modes
- **Camera Views**: Tests different camera configurations
- **Environment Properties**: Verifies metadata and required methods
- **Basic Simulation**: Runs a short simulation to test overall functionality

### 2. `visualization_test.py` - Visual Demonstration
A simple script that provides a visual demonstration of the environment:

- Creates a visual window showing the environment
- Runs a 10-step demonstration with random actions
- Tests different camera views
- Provides detailed output about the environment state

## How to Run the Tests

### Prerequisites
Make sure you have the required dependencies installed:
```bash
pip install gymnasium gymnasium-robotics mujoco stable-baselines3 numpy
```

### Running the Comprehensive Test Suite
```bash
cd franka_mujoco
python test.py
```

This will run all tests and provide a detailed report of what passed or failed.

### Running the Visualization Test
```bash
cd franka_mujoco
python visualization_test.py
```

This will open a visual window showing the environment and run a demonstration.

## Expected Output

### Successful Test Run
```
============================================================
FRANKA PICK AND PLACE ENVIRONMENT RENDERING TESTS
============================================================
Testing environment creation...
✓ Environment created successfully with human render mode
✓ Environment created successfully with rgb_array render mode
✓ Environment created successfully without render mode

Testing environment reset...
✓ Environment reset successful
  - Observation keys: ['observation', 'achieved_goal', 'desired_goal']
  - Info keys: ['is_success']

...

============================================================
TEST SUMMARY: 7/7 tests passed
============================================================
🎉 All tests passed! The environment renders correctly.
```

### Visualization Output
```
Franka Pick and Place Environment Visualization Tests
============================================================
Starting Franka Pick and Place Environment Visualization
==================================================
Resetting environment...
Environment reset successful!
Observation keys: ['observation', 'achieved_goal', 'desired_goal']

Environment Information:
- Action space: Box(-1.0, 1.0, (4,), float32)
- Observation space: Dict('achieved_goal': Box(-inf, inf, (3,), float64), 'desired_goal': Box(-inf, inf, (3,), float64), 'observation': Box(-inf, inf, (25,), float64))
- Render modes: ['human', 'rgb_array']

Running demonstration (10 steps)...
Step 1: Action = [0.2, -0.1, 0.3, 0.5], Reward = -1.0000, Total = -1.0000
...
```

## Troubleshooting

### Common Issues

1. **Import Error**: If you get import errors for `panda_mujoco_gym`, make sure the environment is properly installed or the path is set correctly.

2. **Rendering Issues**: If the visual window doesn't appear, check that:
   - You have a display server running (for headless systems, use `xvfb-run`)
   - MuJoCo is properly installed
   - Your system supports OpenGL rendering

3. **Camera Errors**: Some camera views might not be available depending on the environment configuration. The tests will report which cameras work.

### Running on Headless Systems
If you're running on a system without a display, you can use:
```bash
xvfb-run -a python visualization_test.py
```

### Debug Mode
For more detailed error information, you can modify the test files to include more verbose error handling and logging.

## Test Coverage

The tests cover:
- ✅ Environment creation with different render modes
- ✅ Environment reset functionality
- ✅ Action space and observation space
- ✅ Step function with rewards and termination
- ✅ Rendering in both human and rgb_array modes
- ✅ Multiple camera views
- ✅ Environment metadata and properties
- ✅ Basic simulation functionality
- ✅ Proper cleanup and resource management

## Contributing

When adding new features to the environment, please:
1. Add corresponding tests to `test.py`
2. Update this README if new test functionality is added
3. Ensure all tests pass before submitting changes 