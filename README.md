# Precision-Focused Reinforcment Learning for Robotic Object Pushing
Code for training and sim2real transfer of the reinforcement learning agents from our paper "Precision-Focused Reinforcement Learning Model for Robotic Object Pushing".

<img src="https://github.com/ubi-coro/precise_pushing/raw/main/assets/visual_abstract.png" />

## Installation 
```bash
conda create -n precise_pushing
conda activate precise_pushing
cd PATH_TO_THIS_REPO
pip install -e .
```
## Gymnasium Environments
This repos contains two main Gymnasium environments:
- `MujocoUR3PushSimpleEnv`:
  
  A simple Gymnasium environment mainly used for debugging. An observation consists of the `(x,y)` end-effector position, the `(x,y)` position of the object (achieved goal) and the `(x,y)` position of the target (desired goal).
- `MujocoUR3PushEnv`:
  
  Gymnasium environment that is based on the [vision-proprioception model](https://www.frontiersin.org/articles/10.3389/fnbot.2022.829437/full).
  However, the environment provides more functionality than suggested in the paper (for example objects with variable height and the full episode history of observations).

## Training an Agent
An agent can be trained using the following Python script (Please note that you must first train the autoencoder!):
```bash
python3 PATH_TO_PUSHING_REPO/ur3_push_rl_sb3/train.py
```
For example, to train an agent that automatically adjusts the number of simulation steps use:
```bash
python3 PATH_TO_PUSHING_REPO/ur3_push_rl_sb3/train.py --numSimSteps -1
```
There are many environment and training parameters that can be adjusted. For a complete overview of all configurable parameters use:
```bash
python3 PATH_TO_PUSHING_REPO/ur3_push_rl_sb3/train.py --help
```
In general, evaluation parameters begin with "e", for example "eObjType". They are not used in the training script. 
Log and evaluation files and are saved in the directory `PATH_TO_PUSHING_REPO/ur3_push_data/rl/RUN_NAME`.
`RUN_NAME` is determined by the parameters used to train an agent.
If a parameter differs from its default value, it will appear in RUN_NAME, except for the evaluation parameters (starting with "e") 
and some special parameters that do not influence the training results, for example the log path.

## Evaluation of a Trained Agent
The adjustable parameters are similar to the ones used to train an agent, except that evaluation parameters starting with "e" are not ignored. 
For a complete overview of all configurable parameters use:
```bash
python3 PATH_TO_PUSHING_REPO/ur3_push_rl_sb3/evaluate_policy.py --help
```
In general, parameters not beginning with "e" are used to determine the policy to load, i.e. the training configuration, 
whereas parameters beginning with "e" determine the test configuration.
For example, to test the behavior only for cuboids with square base over 100 evaluation episodes use:
```bash
python3 PATH_TO_PUSHING_REPO/ur3_push_rl_sb3/evaluate_policy.py --eObjType box --eObjSize1 -2 --eNumEvalEpisodes 100
```

## Maintainer
This repository is currently maintained by Lara Bergmann (@lbergmann1).
