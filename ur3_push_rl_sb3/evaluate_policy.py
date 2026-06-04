import os
import numpy as np
import gymnasium as gym
from ur3_push_rl_sb3.utils import parse_args, get_run_name, get_log_paths
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3 import PPO
import logging
import time

# input args that determine dir name of best model
config, cmd_args, config_parser = parse_args()
run_name = get_run_name(config, cmd_args, config_parser)

# log paths
# plots
plot_path = os.path.join(config.logDir, "plots")
if len(config.ePlotName) > 0:
    os.makedirs(plot_path, exist_ok=True)
# videos
video_path = os.path.join(config.logDir, "videos")
if len(config.eVideoName) > 0 and not "Ros" in config.eEnvStr:
    os.makedirs(video_path, exist_ok=True)
# best model
_, eval_path, _, _ = get_log_paths(config.logDir, run_name)

# environment
# object params
assert config.eObjRangeRadius[0] <= config.eObjRangeRadius[1]
assert config.eObjRangeWidthLength[0] <= config.eObjRangeWidthLength[1]
assert config.eObjRangeMass[0] <= config.eObjRangeMass[1]
assert config.eObjRangeSlidingFriction[0] <= config.eObjRangeSlidingFriction[1]
assert config.eObjRangeTorsionalFriction[0] <= config.eObjRangeTorsionalFriction[1]
object_params = {"range_radius": np.array(config.eObjRangeRadius), # [m]
                 "range_wl": np.array(config.eObjRangeWidthLength), # [m]
                 "range_mass": np.array(config.eObjRangeMass), # [kg]
                 "range_sliding_fric": np.array(config.eObjRangeSlidingFriction), 
                 "range_torsional_fric": np.array(config.eObjRangeTorsionalFriction)}
# object reset options
object_reset_options = { 
                        "obj_type": config.eObjType,
                        "obj_mass": config.eObjMass,
                        "obj_sliding_friction": config.eObjSlidingFriction,
                        "obj_torsional_friction": config.eObjTorsionalFriction,
                        "obj_size_0": config.eObjSize0,
                        "obj_size_1": config.eObjSize1,
                        "obj_size_2": config.eObjSize2,
                        "obj_xy_pos": np.array(config.objXYPos, dtype=np.float64) if config.objXYPos is not None else None,
                        "obj_quat": np.array(config.objQuat, dtype=np.float64) if config.objQuat is not None else None,
                        "target_xy_pos": np.array(config.targetXYPos, dtype=np.float64) if config.targetXYPos is not None else None,
                        "target_quat": np.array(config.targetQuat, dtype=np.float64) if config.targetQuat is not None else None,            
                        }
# render mode
if len(config.eVideoName) > 0:
    render_mode = "rgb_array"
else:
    render_mode = "human"
# env_kwargs
env_kwargs = {
        "object_reset_options": object_reset_options,
        "fixed_object_height": config.fixedObjectHeight,
        "sample_mass_slidfric_from_uniform_dist": config.eSampleMassFricFromUniformDist,
        "scale_exponential": config.eScaleExponential,
        "threshold_pos": config.eThresholdPos,
        "sparse_reward": config.sparseReward,
        "n_substeps": -1 if config.numSimSteps == -1 else config.eNumSimSteps,
        "action_scaling_factor": config.eActionScalingFactor
    }
if "Simple" not in config.eEnvStr:
    env_kwargs.update({
        "object_params": object_params,
        "fixed_object_height_VAE": config.eFixedObjectHeightVAE,
        "threshold_zangle": config.eThresholdzAngle,
        "threshold_latent_space": config.eThresholdLatentSpace,
        "ground_truth_dense_reward": config.groundTruthDenseReward,
        "consider_object_orientation": config.eConsiderObjectOrientation,
        "latent_dim": config.latentDim,
        "encode_ee_pos": config.encodeEEPos,
        "use_fingertip_sensor": config.useFingertipSensor,
        "use_obs_history": config.useGRUFeatExtractor,
        "num_stack_obs": config.numStackedObs
    })

env_kwargs.update({"render_mode": render_mode, 
                    "use_sim_config": config.eUseSimConfig,
                    "safety_dq_scale": config.eSafetyDQScale})

# warnings
if config.eUseSimConfig != config.useSimConfig:
    logging.warn(f"controller + camera config training != controller + camera config evaluation")
if config.eSafetyDQScale != config.safetyDQScale:
    logging.warn(f"Safety dq scale training != safety dq scale evaluation")
if config.eConsiderObjectOrientation != config.considerObjectOrientation:
    logging.warn(f"consider object orientation training ({config.considerObjectOrientation}) != consider object orientation evaluation ({config.eConsiderObjectOrientation})")
if config.actionScalingFactor != config.eActionScalingFactor:
    logging.warn(f"Action scaling factor training ({config.actionScalingFactor}) != action scaling factor evaluation ({config.eActionScalingFactor})")
if config.numSimSteps != env_kwargs["n_substeps"]:
    logging.warn(f"Number of simulation steps training ({config.numSimSteps}) != number of simulation steps evaluation ({env_kwargs['n_substeps']})")
if config.thresholdPos != config.eThresholdPos:
    logging.warn(f"Position threshold training ({config.thresholdPos}) != position threshold evaluation ({config.eThresholdPos})")
if config.thresholdzAngle != config.eThresholdzAngle:
    logging.warn(f"Rotation threshold training ({config.thresholdzAngle}) != rotation threshold evaluation ({config.eThresholdzAngle})")
if config.thresholdLatentSpace != config.eThresholdLatentSpace:
    logging.warn(f"Latent space threshold training ({config.thresholdLatentSpace}) != latent space threshold evaluation ({config.eThresholdLatentSpace})")

env = gym.envs.make(config.eEnvStr, **env_kwargs)

# load best model
model = PPO.load(path=os.path.join(eval_path,"best_model"), env=env)
venv = model.get_env()
venv.seed(config.eEvalSeed)

# Record the video starting at the first step
if len(config.eVideoName) > 0 and not "Ros" in config.eEnvStr:
    video_length = config.eNumEvalEpisodes * config.maxEpisodeSteps  # maxEpisodeSteps = 100 by default

    venv = VecVideoRecorder(venv, video_path,
                        record_video_trigger=lambda x: x == 0, video_length=video_length,
                        name_prefix=config.eVideoName)
    print("Starting video recording... This requires offscreen rendering.")

# evaluate
rewards = -2 * np.ones(config.eNumEvalEpisodes)
success = -1* np.ones((config.eNumEvalEpisodes, config.maxEpisodeSteps))
num_corrections = -1* np.ones(config.eNumEvalEpisodes)
num_dist_corrections = -1* np.ones(config.eNumEvalEpisodes)
num_sim_steps = -2 * np.ones((config.eNumEvalEpisodes, config.maxEpisodeSteps), dtype=np.int32)
distance_pos = -1 * np.ones((config.eNumEvalEpisodes, 3))

obs = venv.reset()
for i in range(0,config.eNumEvalEpisodes):
    done = False
    sum_ep_rewards = 0
    step = 0

    if config.eVerbose == 1:
        print(f"Episode {i}:" 
                f"\n\tobjType:\t\t{venv.get_attr(attr_name='obj_type')[0]}" \
                f"\n\tobjStartPos:\t\t{venv.get_attr(attr_name='obj_start_pos')[0]}" \
                f"\n\tobjStartQuat:\t\t{venv.get_attr(attr_name='obj_start_quat')[0]}" \
                f"\n\tobjMass:\t\t{venv.get_attr(attr_name='obj_mass')[0]:.4f}" \
                f"\n\tobjSlidingFriction:\t{venv.get_attr(attr_name='obj_sliding_fric')[0]:.4f}" \
                f"\n\tobjTorsionalFriction:\t{venv.get_attr(attr_name='obj_torsional_fric')[0]:.4f}" \
                f"\n\tobjSize0:\t\t{venv.get_attr(attr_name='obj_size_0')[0]:.4f}" \
                f"\n\tobjSize1:\t\t{venv.get_attr(attr_name='obj_size_1')[0]:.4f}" \
            )
        if venv.get_attr(attr_name="obj_type")[0] == "box":
            print(f"\tobjSize2:\t\t{venv.get_attr(attr_name='obj_size_2')[0]:.4f}")
    
    while not done:
        action, state = model.predict(obs, deterministic=config.eDeterministicEvalPolicy)
        obs, reward, done, info  = venv.step(action) # DummyVecEnv resets env automatically
        sum_ep_rewards += reward[0]

        if config.eStepDelay > 0:
            time.sleep(config.eStepDelay)
        if config.eStepByStep:
            input(f"Step {step} completed. Press Enter to continue...")

        success[i,step] = info[0]["is_success"]
        num_sim_steps[i,step] = action[0,-1] if config.numSimSteps == -1 else config.numSimSteps
        step += 1
        if step == 1:
            # distance at the beginning of an episode (the end of time step 0)
            distance_pos[i,0] = info[0]["dist_pos"] 
        elif step == 25:
            # distance at the end of time step 25
            distance_pos[i,1] = info[0]["dist_pos"]

    # distance at the end of an episode
    distance_pos[i,2] = info[0]["dist_pos"]

    rewards[i] = sum_ep_rewards
    if "num_corrective_movements" in info[0].keys():
        num_corrections[i] = info[0]["num_corrective_movements"]
        num_dist_corrections[i] = info[0]["num_distance_corrections"]

        if config.eVerbose == 1:
            print(f"\tnum_corrections:\t{int(num_corrections[i])}")
            print(f"\tnum_dist_corrections:\t{int(num_dist_corrections[i])}")

        if num_corrections[i] > num_dist_corrections[i]:
            assert num_corrections[i]-1 == num_dist_corrections[i]
            assert success[i,-1] == False
    if config.eVerbose == 1:
        print(f"\tsuccess:\t\t{bool(success[i,-1])}\n")
    
print()
print(f"success rate: {np.mean(success[:,-1])*100:.2f} +/- {np.std(success[:,-1])*100:.2f}")
print(f"mean_reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
if "num_corrective_movements" in info[0].keys():
    print(f"mean num_corrections: {np.mean(num_corrections):.2f} +/- {np.std(num_corrections):.2f}, min num_corrections: {int(np.min(num_corrections))}, max num_corrections: {int(np.max(num_corrections))}")
    print(f"mean num_dist_corrections: {np.mean(num_dist_corrections):.2f} +/- {np.std(num_dist_corrections):.2f}, min num_dist_corrections: {int(np.min(num_dist_corrections))}, max num_dist_corrections: {int(np.max(num_dist_corrections))}")
print()

