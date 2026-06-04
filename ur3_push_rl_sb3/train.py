#!/usr/bin/env python
import os, sys, json, pickle, shutil
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnMaxEpisodes
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from ur3_push_rl_sb3.make_envs import make_vec_envs, make_warp_env
from ur3_push_rl_sb3.utils import (parse_args, 
                                     get_run_name, 
                                     get_log_paths, 
                                     close_envs, 
                                     get_system_info_dict, 
                                     linear_lr_schedule)
from ur3_push_rl_sb3.custom_callbacks import CustomCheckpointCallback
from stable_baselines3.common.torch_layers import CombinedExtractor
from ur3_push_rl_sb3.custom_features_extractors import GRUExtractor

def main():
    config, cmd_args, config_parser = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # log paths
    eval_dir_name = "evaluation"
    log_dir_name = "logs"
    cp_dir_name = "checkpoint"
    save_path, eval_path, log_path, cp_path = get_log_paths(config.logDir, 
                                                            get_run_name(config, cmd_args, config_parser),
                                                            eval_dir_name,
                                                            log_dir_name,
                                                            cp_dir_name)
    
    # continue training?
    continue_train = ""
    reset_num_timesteps = True # new training curves in tensorboard
    rng_states_envs = None # set train/test seed instead if loading RNG states
    override_monitor_logs = True # append logs to existing monitor log files?
    if os.path.exists(log_path):
        print(f"\nlog path already exists. Current log path is: {save_path}")
        while continue_train not in ["y", "n", "r"]:
            continue_train = input("Continue training (y), Restart and overwrite (r), or Cancel (n)? ")
    
        if continue_train == "n":
            print("No files have been changed.")
            sys.exit()
        elif continue_train == "r":
            print(f"Overwriting existing log path: {save_path}")
            shutil.rmtree(save_path)
            os.makedirs(save_path, exist_ok=True)
        elif not os.path.exists(cp_path):
            # continue_train == "y", but no checkpoint exists
            print("Cannot continue training: log path exists, but no checkpoint found\n")
            sys.exit()
        else:
            # continue_train == "y" and checkpoint exists
            reset_num_timesteps = False # continue training curves in tensorboard
            override_monitor_logs = False
            # reset log and evaluation files 
            shutil.rmtree(path=log_path)
            shutil.rmtree(path=eval_path)
            shutil.copytree(src=os.path.join(cp_path, log_dir_name), dst=log_path)
            shutil.copytree(src=os.path.join(cp_path, eval_dir_name), dst=eval_path)
            # load RNG states
            with open(os.path.join(cp_path,"rng_states_gymenvs.pkl"), mode="rb") as rng_env_file:
                rng_states_envs = pickle.load(rng_env_file)
    
    try:
        # object reset options
        object_reset_options = { 
                                "obj_type": config.objType,
                                "obj_mass": config.objMass,
                                "obj_sliding_friction": config.objSlidingFriction,
                                "obj_torsional_friction": config.objTorsionalFriction,
                                "obj_size_0": config.objSize0,
                                "obj_size_1": config.objSize1,
                                "obj_size_2": config.objSize2,
                                "obj_xy_pos": np.array(config.objXYPos, dtype=np.float64) if config.objXYPos is not None else None,
                                "obj_quat": np.array(config.objQuat, dtype=np.float64) if config.objQuat is not None else None,
                                "target_xy_pos": np.array(config.targetXYPos, dtype=np.float64) if config.targetXYPos is not None else None,
                                "target_quat": np.array(config.targetQuat, dtype=np.float64) if config.targetQuat is not None else None,            
                                }
        
        # env_kwargs
        env_kwargs = {
                "object_reset_options": object_reset_options,
                "fixed_object_height": config.fixedObjectHeight,
                "sample_mass_slidfric_from_uniform_dist": config.sampleMassFricFromUniformDist,
                "scale_exponential": config.scaleExponential,
                "threshold_pos": config.thresholdPos,
                "sparse_reward": config.sparseReward,
                "n_substeps": config.numSimSteps,
                "action_scaling_factor": config.actionScalingFactor
            }
        if "Simple" not in config.envStr:
            env_kwargs.update({
                "fixed_object_height_VAE": config.fixedObjectHeight,
                "threshold_zangle": config.thresholdzAngle,
                "threshold_latent_space": config.thresholdLatentSpace,
                "ground_truth_dense_reward": config.groundTruthDenseReward,
                "consider_object_orientation": config.considerObjectOrientation,
                "latent_dim": config.latentDim,
                "encode_ee_pos": config.encodeEEPos,
                "use_fingertip_sensor": config.useFingertipSensor,
                "use_obs_history": config.useGRUFeatExtractor,
                "num_stack_obs": config.numStackedObs
            })
        else:
            # SimpleEnv-specific params
            env_kwargs.update({
                "ee_to_obj_reward_scale": config.eeToObjRewardScale
            })
    
        env_kwargs.update({"render_mode": "rgb_array",
                            "use_sim_config": config.useSimConfig,
                            "safety_dq_scale": config.safetyDQScale}) 
        vec_env_cls = SubprocVecEnv if config.numTrain > 1 else DummyVecEnv
        env_has_id = False
    
        # make envs
        if getattr(config, 'useWarp', 0):
            # --- MuJoCo Warp: GPU batch (N mundos em paralelo) ---
            print(f"[WarpVecEnv] Usando MuJoCo Warp com nworld={config.numTrain} mundos na GPU")
            train_envs, eval_env = make_warp_env(
                num_train=config.numTrain,
                max_episode_steps=config.maxEpisodeSteps,
                sparse_reward=bool(config.sparseReward),
                ee_to_obj_reward_scale=config.eeToObjRewardScale,
                action_scaling_factor=config.actionScalingFactor,
                n_substeps=config.numSimSteps,
                seed=config.trainSeed,
            )
        else:
            # --- CPU: SubprocVecEnv (comportamento original) ---
            train_envs, eval_env = make_vec_envs(config.envStr, 
                                                env_has_id,
                                                log_path, 
                                                vec_env_cls,
                                                config.numTrain, 
                                                config.trainSeed, 
                                                config.evalSeed, 
                                                rng_states_envs,
                                                override_monitor_logs,
                                                max_episode_steps=config.maxEpisodeSteps,
                                                **env_kwargs)
    
    
        if continue_train in ["", "r"]: # train new model
            # garante que o diretório de logs existe (WarpVecEnv não cria via Monitor)
            os.makedirs(log_path, exist_ok=True)
            # save config
            with open(os.path.join(log_path,"config.txt"),"w") as f:
                json.dump(config.__dict__, f, indent=2)
            
            # save system info
            system_info_dict = get_system_info_dict()
            with open(os.path.join(log_path, "system_info.txt"), "w") as f:
                json.dump(system_info_dict, f, indent=2)
    
            # policy
            policy_kwargs = dict(
                                share_features_extractor=config.shareFeatExtractor,
                                net_arch=config.policyNetArch)
    
            if config.useGRUFeatExtractor:
                policy_kwargs.update(dict(
                                features_extractor_class = GRUExtractor,
                                features_extractor_kwargs = {"features_dim": config.GRUFeaturesDim}
                ))
            else:
                policy_kwargs.update(dict(features_extractor_class = CombinedExtractor))
    
            # learning rate
            if config.useLinearlrSchedule:
                lr = linear_lr_schedule(initial_value=config.ppolr)
            else:
                lr = config.ppolr
    
            # PPO
            model = PPO(
                policy = "MultiInputPolicy",
                env = train_envs,
                learning_rate = lr,
                n_steps = config.nSteps,
                batch_size = config.batchSize,
                n_epochs = config.nEpochs,
                gamma = config.gamma,
                clip_range = config.clipRange,
                ent_coef = config.entCoef,
                tensorboard_log=log_path, # log location of tensorboard (if None, no logging)
                policy_kwargs = policy_kwargs,
                verbose=1,
                seed=config.trainSeed, # this also sets seed of envs
                device=device
            )
    
        else:
            # continue_train == "y"
            model = PPO.load(path=os.path.join(cp_path,"model"), env=train_envs)
    
        # logger
        logger = configure(log_path, ["stdout", "csv", "tensorboard"])
        model.set_logger(logger)
    
        # callbacks
        stop_train_cb = StopTrainingOnMaxEpisodes(max_episodes=config.maxTrainEpisodes, verbose=1)
        # Para Warp, evalFreq ja foi calculado com base em numTrain no utils.py,
        # e o EvalCallback conta em gym steps (cada step = 1 no WarpVecEnv, mas envolve N mundos).
        # Nao dividir novamente por numTrain para evitar avaliacao a cada step.
        if getattr(config, 'useWarp', 0):
            eval_freq_steps = max(1, config.evalFreq)  # ja em gym steps
        else:
            eval_freq_steps = max(1, int(config.evalFreq / config.numTrain))
        eval_cb = EvalCallback( eval_env, 
                                best_model_save_path=eval_path,
                                log_path=eval_path, 
                                eval_freq=eval_freq_steps,
                                n_eval_episodes=config.nEvalEpisodes,
                                deterministic=config.determinsticEvalPolicy)
        checkpoint_cb = CustomCheckpointCallback(
                                calls_before_saving=1, # save after the first eligible rollout (evalFreq was wrong unit: steps vs rollouts)
                                save_freq=config.saveFreq,
                                save_path=save_path,
                                cp_path=cp_path,
                                train_envs=train_envs,
                                eval_env=eval_env,
                                log_dir_name=log_dir_name,
                                eval_dir_name=eval_dir_name,
                                verbose=2)
        callbacks = CallbackList([eval_cb, checkpoint_cb, stop_train_cb])
            
        # train model
        model.learn(total_timesteps=config.totalLearningTimesteps - model.num_timesteps,
                    callback=callbacks,
                    log_interval=config.numTrain,
                    reset_num_timesteps=reset_num_timesteps,
                    progress_bar=True)
        
    except KeyboardInterrupt:
        pass
    except Exception:
        close_envs(train_envs, eval_env, config)
        raise
    else:
        close_envs(train_envs, eval_env, config)

if __name__ == '__main__':
    main()
