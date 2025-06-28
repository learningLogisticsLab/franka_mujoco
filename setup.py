from setuptools import setup, find_packages

setup(
    name="panda_mujoco_gym",
    version="0.1",
    description="Franka Mujoco Environemnts for Reinforcement Learning",
    url="https://github.com/LearningLogisicsLab/panda_mujoco_gym",
    author="auth",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(
        include=[
            "panda_mujoco_gym",
        ]
    ),
    install_requires=[
        "gymnasium>=0.29.1",                     # or gym, depending on what you use
        "mujoco>=2.3.3",
        # list any other runtime dependencies here
    ],
    zip_safe=False,
)
