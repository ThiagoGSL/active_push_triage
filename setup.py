import pathlib

from setuptools import setup, find_packages

CWD = pathlib.Path(__file__).absolute().parent

setup(
    name="precise_pushing",
    packages=[package for package in find_packages() if package.startswith("ur3_push")],
    install_requires = ["numpy>=1.20,<1.24.0",
                        "matplotlib>=3.3,!=3.6.1",
                        "seaborn",
                        "opencv-python",
                        "gymnasium-robotics",
                        "stable_baselines3@git+https://github.com/lbergmann1/stable-baselines3@devel", # stable-baselines3 with Gymnasium support, save/load internal RNG states and relabel info["desired_goal"]
                        "mujoco==2.3.3",
                        "requests>=2.28.2",
                        "tensorboard",
                        "torch>=2.0", 
                        "gymnasium==0.28.1",
                        "tqdm", # stable-baselines3 progress bar callback
                        "rich", # stable-baselines3 progress bar callback
                        "ray[default]",
                        "moviepy"
                        ],
    entry_points={"gymnasium.envs":["__root__ = ur3_push_mujoco.gym_ur3_push.__init__:register_gymnasium_envs"]},
    description="Precise planar object pushing using reinforcement learning based on MuJoCo and ROS", 
    author="Lara Bergmann",
    author_email="lara.bergmann@uni-bielefeld.de",
    url="https://github.com/ubi-coro/precise_pushing/",
    python_requires=">=3.8",
)
