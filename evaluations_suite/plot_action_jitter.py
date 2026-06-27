import os
import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecFrameStack

import sys
sys.path.append(r"c:\Projetos\TCC\active_push_triage")
from ur3_push_rl_sb3.make_envs import make_warp_env

def load_policy(eval_path, algorithm):
    config_path = os.path.join(eval_path, "logs", "config.txt")
    with open(config_path, "r") as f:
        train_config = json.load(f)

    _, eval_env = make_warp_env(
        num_train=1,
        max_episode_steps=train_config.get("maxEpisodeSteps", 100),
        sparse_reward=bool(train_config.get("sparseReward", 0)),
        ee_to_obj_reward_scale=0.0,
        action_scaling_factor=train_config.get("actionScalingFactor", 0.05),
        n_substeps=train_config.get("numSimSteps", 120),
        seed=42, # Fixed seed
        torque_penalty_scale=train_config.get("torquePenaltyScale", 0.001),
        action_rate_penalty_scale=train_config.get("actionRatePenaltyScale", 0.05),
        success_bonus=train_config.get("successBonus", 0.0),
        early_termination_on_success=0,
        randomize_initial_joints=bool(train_config.get("randomizeInitialJoints", 0)),
        manipulability_reward_scale=train_config.get("manipulabilityRewardScale", 0.01),
        manipulability_metric=train_config.get("manipulabilityMetric", "yoshikawa")
    )
    
    warp_wenv = eval_env
    num_stacked = train_config.get("numStackedObs", 4)
    if num_stacked > 1:
        eval_env = VecFrameStack(eval_env, n_stack=num_stacked)

    model_path = os.path.join(eval_path, "evaluation", "best_model")
    if not os.path.exists(model_path + ".zip"):
        model_path = os.path.join(eval_path, "best_model")
        
    if algorithm.lower() == "sac":
        model = SAC.load(model_path, env=eval_env)
    else:
        model = PPO.load(model_path, env=eval_env)
        
    return model, eval_env, warp_wenv

def extract_action_trajectory(model, eval_env, warp_wenv):
    class PositionSampler:
        def __init__(self):
            self.call_count = 0
            
        def __call__(self, ref_xy, min_dist):
            is_target = (self.call_count % 2) == 1
            self.call_count += 1
            if is_target:
                return np.array([0.45, 0.15]) # Target Pos
            else:
                return np.array([0.30, -0.15]) # Obj Pos

    warp_wenv._wenv._sample_xy_away_from = PositionSampler()
    
    obs = eval_env.reset()
    action_traj = []
    
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, dones, infos = eval_env.step(action)
        action_traj.append(action[0].copy())
        
    return np.array(action_traj) # Shape: [100, 6]

if __name__ == "__main__":
    base_data_path = r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl"

    print("Evaluating SAC Actions...")
    sac_path = os.path.join(base_data_path, "MujocoUR3PushEnv_sac_teste_fps_mesa_20260626_015332")
    sac_model, sac_env, sac_wenv = load_policy(sac_path, "sac")
    sac_actions = extract_action_trajectory(sac_model, sac_env, sac_wenv)

    print("Evaluating PPO Actions...")
    ppo_path = os.path.join(base_data_path, "MujocoUR3PushEnv_ppo_teste_fps_mesa_20260625_223152")
    ppo_model, ppo_env, ppo_wenv = load_policy(ppo_path, "ppo")
    ppo_actions = extract_action_trajectory(ppo_model, ppo_env, ppo_wenv)

    # Calculate Jitter: norm of difference between consecutive actions
    sac_jitter = np.linalg.norm(np.diff(sac_actions, axis=0), axis=1)
    ppo_jitter = np.linalg.norm(np.diff(ppo_actions, axis=0), axis=1)

    print("Plotting Jitter...")
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Suavidade de Controle: Jitter da Ação (SAC vs PPO)', fontsize=16)

    ax.plot(range(1, 100), sac_jitter, 'r-', label='SAC', linewidth=2, alpha=0.8)
    ax.plot(range(1, 100), ppo_jitter, 'b-', label='PPO', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Passo da Simulação', fontsize=12)
    ax.set_ylabel('Jerk (Norma da diferença da Ação)', fontsize=12)
    ax.set_title(f'Média de Jitter - SAC: {np.mean(sac_jitter):.4f} | PPO: {np.mean(ppo_jitter):.4f}', fontsize=14)
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    output_img = os.path.join(os.path.dirname(__file__), "results", "action_jitter.png")
    fig.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Saved {output_img}")
