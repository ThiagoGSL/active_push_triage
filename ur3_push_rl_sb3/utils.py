import argparse, sys, os
import cv2 as cv
import tensorboard, ray, mujoco
from stable_baselines3.common.utils import get_system_info
from copy import deepcopy

def parse_args():
    ############################################
    # train/eval params
    ############################################
    train_eval_parser = argparse.ArgumentParser(add_help=False)
    train_eval_group = train_eval_parser.add_argument_group("train_eval", "general train and eval params")
    train_eval_group.add_argument("--numTrain", type=int, default=1, help="number of parallel training environments")
    train_eval_group.add_argument("--trainSeed", type=int, default=1, help="training seed (train enviroment with id i has training seed trainSeed+i)")
    train_eval_group.add_argument("--evalSeed", type=int, default=42, help="seed used for evaluation environment during training")
    train_eval_group.add_argument("--envStr", type=str, default="MujocoUR3PushEnv", help="Gymnasium env id (str)")
    train_eval_group.add_argument("--useFingertipSensor", type=int, choices=[0, 1], default=0, help="use fingertip sensor data? (only used if ee='fingertip')")
    train_eval_group.add_argument("--useSimConfig", type=int, choices=[0, 1], default=1, help="use sim or real controller + camera config? (real controller config fixes some problems with the real robot controller")

    train_eval_config, _ = train_eval_parser.parse_known_args()

    config_parser = argparse.ArgumentParser(parents=[train_eval_parser])
    ############################################
    # params gym env
    ############################################
    gym_env_group = config_parser.add_argument_group("gymnasium environments", "params gymnasium environment (params are used for all train and test envs)")
    gym_env_group.add_argument("--sparseReward", type=int, choices=[0, 1], default=1, help="use sparse reward?")
    gym_env_group.add_argument("--groundTruthDenseReward", type=int, choices=[0,1], default=0, help="use ground truth dense reward? (only used if ee='fingertip')")
    gym_env_group.add_argument("--considerObjectOrientation", type=int, choices=[0, 1], default=0, help="consider object orientation in reward function?")
    gym_env_group.add_argument("--actionScalingFactor", type=float, default=1.0, help="x,y actions are multiplied by this factor (not used in case of action scaling env)")
    gym_env_group.add_argument("--numSimSteps", type=int, default=120, help="number of simulation steps that are executed if gym.step() is called (-1: RL agent learns number of sim steps)")
    gym_env_group.add_argument("--safetyDQScale", type=float, default=1.0, help="Safety scaling factor for desired velocities")
    gym_env_group.add_argument("--latentDim", type=lambda x: None if x == 'None' else int(x), default=6, help="number of VAE latent dimensions")
    gym_env_group.add_argument("--fixedObjectHeight", type=lambda x: None if x == 'None' else float(x), default=0.023, help="Use fixed object height? No: None, Yes: float; Training: uses a VAE was trained with the same value")
    gym_env_group.add_argument("--encodeEEPos", type=int, choices=[0, 1], default=0, help="use latent ee state in observation? (not used if envStr='MujocoUR3PushSimpleEnv')")
    gym_env_group.add_argument("--thresholdPos", type=float, default=0.01, help="position threshold [m]")
    gym_env_group.add_argument("--thresholdzAngle", type=float, default=10, help="rotation threshold [deg])")
    gym_env_group.add_argument("--thresholdLatentSpace", type=float, default=0.15, help="threshold in latent space (not used if envStr='MujocoUR3PushSimpleEnv')")
    gym_env_group.add_argument("--sampleMassFricFromUniformDist", type=int, choices=[0, 1], default=1, help="whether to sample object mass and sliding friction coefficient from continuous unifrom distribution (if 0: sample from modified exponential distribution if obj mass and friction param are sampled)")
    gym_env_group.add_argument("--scaleExponential", type=float, default=1/7, help="scale_exponential: sclae of the exponential distribution used to sample mass and sliding fricition coefficient if sample_mass_slidfric_from_uniform_dist = 0")
    gym_env_group.add_argument("--numStackedObs", type=lambda x: None if x == 'None' else int(x), default=None, help="Number of stacked observations; use None for no stacking")
    # desired object/target params 
    gym_env_group.add_argument("--objType", type=lambda x: None if x == 'None' else str(x), default=None, help="object type; None (default): random, '': initial value is not changed, 'box' or 'cylinder'")
    gym_env_group.add_argument("--objMass", type=lambda x: None if x == 'None' else float(x), default=None, help="object mass; None (default): random, -1: initial value is not changed")
    gym_env_group.add_argument("--objSlidingFriction", type=lambda x: None if x == 'None' else float(x), default=None, help="object sliding friction; None (default): random, -1: initial value is not changed")
    gym_env_group.add_argument("--objTorsionalFriction", type=lambda x: None if x == 'None' else float(x), default=None, help="object torsional friction; None (default): random, -1: initial value is not changed")
    gym_env_group.add_argument("--objSize0", type=lambda x: None if x == 'None' else float(x), default=None, help="box: obj_width/2, cylinder: obj_radius; None (default): random, -1: initial value is not changed")
    gym_env_group.add_argument("--objSize1", type=lambda x: None if x == 'None' else float(x), default=None, help="box: obj_length/2, cylinder: obj_height/2; None (default): random, -1: initial value is not changed, -2 and box: length=width")
    gym_env_group.add_argument("--objSize2", type=lambda x: None if x == 'None' else float(x), default=None, help="box: obj_height/2, cylinder: ignored; None (default): random, -1: initial value is not changed")
    gym_env_group.add_argument("--objXYPos", nargs=2, type=lambda x: None if x == 'None' else float(x), default=None, help="xy start pos of the object; None (default): random")
    gym_env_group.add_argument("--objQuat", nargs=4, type=lambda x: None if x == 'None' else float(x), default=None, help="start orientation of the object (x,y,z,w); None (default): random")
    gym_env_group.add_argument("--targetXYPos", nargs=2, type=lambda x: None if x == 'None' else float(x), default=None, help="xy target pos of the object; None (default): random")
    gym_env_group.add_argument("--targetQuat", nargs=4, type=lambda x: None if x == 'None' else float(x), default=None, help="target orientation of the object (x,y,z,w); None (default): random")

    ############################################
    # params PPO
    ############################################
    ppo_group = config_parser.add_argument_group("PPO", "params Proximal Policy Optimization")
    ppo_group.add_argument("--ppolr", type=float, default=3e-4, help="PPO learning rate")
    ppo_group.add_argument("--useLinearlrSchedule", type=int, default=1, help="Wether to use a linear learning rate schedule. initial value: ppolr.")
    ppo_group.add_argument("--nSteps", type=int, default=2048, help="number of steps to run for each environment per update")
    ppo_group.add_argument("--batchSize", type=int, default=64, help="minibatch size")
    ppo_group.add_argument("--nEpochs", type=int, default=10, help="number of epochs when optimizing the surrogate loss")
    ppo_group.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    ppo_group.add_argument("--clipRange", type=float, default=0.2, help="clipping parameter for the value function")
    ppo_group.add_argument("--entCoef", type=float, default=0.0, help="entropy regularization coefficient")
    ppo_group.add_argument("--policyNetArch", nargs="+", type=int, default=[128,256,64], help="size of MLP used for actor and critic")
    ppo_group.add_argument("--shareFeatExtractor", type=int, default=0, help="whether to share features extractor between actor and critic)")
    ppo_group.add_argument("--useGRUFeatExtractor", type=int, choices=[0, 1], default=0, help="whether to use a custom GRU feature extractor (otherwise: SB3 CombinedExtractor is used)")
    ppo_group.add_argument("--GRUFeaturesDim", type=int, default=32, help="number of features in hidden state (only used if custom GRU feature extractor is used)")
    ppo_group.add_argument("--totalLearningTimesteps", type=int, default=int(3e6), help="max number of learning timesteps")

    ############################################
    # callback config
    ############################################
    cb_group = config_parser.add_argument_group("SB3 callbacks", "params SB3 callbacks")
    cb_group.add_argument("--logDir", type=str, default=os.getenv("UR3_PUSH_DATAPATH"), help="directory where to save log files and best model")
    cb_group.add_argument("--commentLogPath", type=str, default="", help="Comment added to log directory")
    cb_group.add_argument("--maxTrainEpisodes", type=int, default=580, help="callback: StopTrainingOnMaxEpisodes: stops training after maxTrainEpisodes * num_train episodes regardless of totalLearningTimesteps")
    cb_group.add_argument("--evalFreq", type=int, default=int(5000/train_eval_config.numTrain), help="callback: EvalCallback; evaluate model after num_train*eval_freq steps")
    cb_group.add_argument("--saveFreq", type=int, default=int(10000/train_eval_config.numTrain), help="callback: CustomCheckpointCallback; save model after num_train*save_freq steps")
    cb_group.add_argument("--nEvalEpisodes", type=int, default=100, help="callback: EvalCallback; number of episodes to test the agent")
    cb_group.add_argument("--determinsticEvalPolicy", type=int, choices=[0, 1], default=1, help="callback: EvalCallback; whether to use deterministic actions")

    ############################################
    # evaluation config (best model)
    ############################################
    eval_group = config_parser.add_argument_group("Best model evaluation", "params best model evaluation (not used for training)")
    eval_group.add_argument("--eEnvStr", type=str, default="MujocoUR3PushEnv", help="Gymnasium env id (str) used t evaluate the best model")
    eval_group.add_argument("--eUseSimConfig", type=int, choices=[0, 1], default=1, help="use sim or real controller + camera config? (real controller config fixes some problems with the real robot controller")
    eval_group.add_argument("--eConsiderObjectOrientation", type=int, choices=[0, 1], default=0, help="consider object orientation in reward function?")
    eval_group.add_argument("--eDeterministicEvalPolicy", type=int, choices=[0, 1], default=1, help="whether to use deterministic actions")
    eval_group.add_argument("--eNumEvalEpisodes", type=int, default=100, help="number of evaluation episodes")
    eval_group.add_argument("--eActionScalingFactor", type=float, default=1.0, help="x,y actions are multiplied by this factor (not used in case of action scaling env)")
    eval_group.add_argument("--eNumSimSteps", type=int, default=120, help="number of simulation steps that are executed if gym.step() is called")
    eval_group.add_argument("--eSafetyDQScale", type=float, default=1.0, help="Safety scaling factor for desired velocities")
    eval_group.add_argument("--eObjType", type=lambda x: None if x == 'None' else str(x), default=None, help="object type; None (default): random, '': initial value is not changed, 'box' or 'cylinder'")
    eval_group.add_argument("--eObjMass", type=lambda x: None if x == 'None' else float(x), default=None, help="object mass; None (default): random, -1: initial value is not changed")
    eval_group.add_argument("--eObjSlidingFriction", type=lambda x: None if x == 'None' else float(x), default=None, help="object sliding friction; None (default): random, -1: initial value is not changed")
    eval_group.add_argument("--eObjTorsionalFriction", type=lambda x: None if x == 'None' else float(x), default=None, help="object torsional friction; None (default): random, -1: initial value is not changed")
    eval_group.add_argument("--eObjSize0", type=lambda x: None if x == 'None' else float(x), default=None, help="box: obj_width/2, cylinder: obj_radius; None (default): random, -1: initial value is not changed")
    eval_group.add_argument("--eObjSize1", type=lambda x: None if x == 'None' else float(x), default=None, help="box: obj_length/2, cylinder: obj_height/2; None (default): random, -1: initial value is not changed, -2 and box: length=width, -3 and box: abs(length - width)>1e-2")
    eval_group.add_argument("--eObjSize2", type=lambda x: None if x == 'None' else float(x), default=None, help="box: obj_height/2, cylinder: ignored; None (default): random, -1: initial value is not changed")
    eval_group.add_argument("--eFixedObjectHeightVAE", type=lambda x: None if x == 'None' else float(x), default=0.023, help="Use VAE trained with fixed object height for evaluation? No: None, Yes: float")
    eval_group.add_argument("--eThresholdPos", type=float, default=0.01, help="position threshold [m]")
    eval_group.add_argument("--eThresholdzAngle", type=float, default=10, help="rotation threshold [deg])")
    eval_group.add_argument("--eThresholdLatentSpace", type=float, default=0.15, help="threshold in latent space")
    eval_group.add_argument("--eEvalSeed", type=int, default=17, help="seed used to evaluate best model")
    eval_group.add_argument("--eSampleMassFricFromUniformDist", type=int, choices=[0, 1], default=1, help="whether to sample object mass and sliding friction coefficient from continuous unifrom distribution (if 0: sample from modified exponential distribution if obj mass and friction param are sampled)")
    eval_group.add_argument("--eScaleExponential", type=float, default=1/7, help="scale_exponential: sclae of the exponential distribution used to sample mass and sliding fricition coefficient if sample_mass_slidfric_from_uniform_dist = 0")
    eval_group.add_argument("--eObjRangeRadius", nargs=2, type=float, default=[0.08/2, 0.11/2], help="range cylinder radius [m]")
    eval_group.add_argument("--eObjRangeWidthLength", nargs=2, type=float, default=[0.05/2, 0.11/2], help="range width/2, length/2 [m]")
    eval_group.add_argument("--eObjRangeMass", nargs=2, type=float, default=[0.001, 1.0], help="range object mass [kg]")
    eval_group.add_argument("--eObjRangeSlidingFriction", nargs=2, type=float, default=[0.2, 1.0], help="range sliding friction coefficient")
    eval_group.add_argument("--eObjRangeTorsionalFriction", nargs=2, type=float, default=[0.001, 0.01], help="range torsional friction coefficient")
    eval_group.add_argument("--eVerbose", type=int, choices=[0,1], default=1, help="Verbosity level policy evaluation; 0: no output, 1: prints information about object properties and evaluation results of individual episodes")
    eval_group.add_argument("--ePlotName", type=str, default="", help="name of the plots; no plot is saved if empty")
    eval_group.add_argument("--eVideoName", type=str, default="", help="name of the video; no video is saved if empty")
    eval_group.add_argument("--eStepDelay", type=float, default=0.0, help="delay between steps [s]")
    eval_group.add_argument("--eStepByStep", type=int, choices=[0,1], default=0, help="whether to wait for user input after each step")

    config = config_parser.parse_args() # experiment configurations
    cmd_args = sys.argv[1:]
    
    return config, cmd_args, config_parser

def get_run_name(config, cmd_args, config_parser):
    config_tmp = deepcopy(config)
    # append params != default to suffix
    suffix_log_path = ""
    exclude = [ "numTrain", "ee", "envStr", "launchArgs", "maxTrainEpisodes", \
                "logDir", "commentLogPath", "eEnvStr", "eUseSimConfig", "eConsiderObjectOrientation", "eDeterministicEvalPolicy", "eNumEvalEpisodes", "eActionScalingFactor", \
                "eNumSimSteps", "eSafetyDQScale", "eObjType", "eObjMass", "eObjSlidingFriction", "eObjTorsionalFriction", "eObjSize0", \
                "eObjSize1", "eObjSize2", "eFixedObjectHeightVAE", "eThresholdPos", "eThresholdzAngle", "eThresholdLatentSpace", \
                "eEvalSeed", "eSampleMassFricFromUniformDist", "eScaleExponential", "eObjRangeRadius", "eObjRangeWidthLength", "eObjRangeMass", \
                "eObjRangeSlidingFriction", "eObjRangeTorsionalFriction", "eVerbose", "ePlotName", "eVideoName", "eStepDelay", "eStepByStep"]

    arg_name_idx = [i for i in range(0, len(cmd_args)) if cmd_args[i].startswith("--")]
    arg_name_idx.reverse()
    for i in arg_name_idx:
        arg_name = cmd_args[i][2:]
        if arg_name not in exclude:
            default = str(config_parser.get_default(arg_name))
            val = ','.join(cmd_args[i+1:])

            if default.startswith("["): 
                default = default[1:-1].replace(" ","")

            if default != val:
                suffix_log_path = f"_{arg_name}_{val}" + suffix_log_path
        cmd_args = cmd_args[0:i]

    # add comment
    if config_tmp.commentLogPath != "":
        if not config_tmp.commentLogPath.startswith("_"):
            comment = "_" + config_tmp.commentLogPath
        else:
            comment = config_tmp.commentLogPath
        suffix_log_path += comment

    run_name = f"{config_tmp.envStr}{suffix_log_path}"

    return run_name

def get_log_paths(logDir, run_name, eval_dir_name = "evaluation", log_dir_name = "logs", cp_dir_name = "checkpoint"):
    save_path = os.path.join(logDir, "rl", run_name)
    eval_path = os.path.join(save_path, eval_dir_name) # path to a folder where the evaluations and best model will be saved
    log_path = os.path.join(save_path, log_dir_name) # tensorboard logs, monitor logs
    cp_path = os.path.join(save_path, cp_dir_name) # checkpoint path 

    return save_path, eval_path, log_path, cp_path

def close_envs(train_envs, eval_env, config):
    train_envs.close()
    if eval_env is not None:
        eval_env.close()

def get_system_info_dict():
    env_info, _ = get_system_info(print_info=False)
    env_info.update({"OpenCV" : cv.__version__,
                    "Tensorboard": tensorboard.__version__,
                    "Ray" : ray.__version__, 
                    "MuJoCo": mujoco.__version__})
    return env_info

def linear_lr_schedule(initial_value):
    """
    Linear learning rate schedule. (adapted from SB3)

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining):
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value
    
    return func
