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

def extract_manipulability(model, eval_env, warp_wenv):
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
    yoshikawa_traj = []
    
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, dones, infos = eval_env.step(action)
        
        site_xpos = warp_wenv._wenv._d_gpu.site_xpos.numpy()[0]
        xaxis = warp_wenv._wenv._d_gpu.xaxis.numpy()[0, :6]
        xanchor = warp_wenv._wenv._d_gpu.xanchor.numpy()[0, :6]
        
        ee_pos = site_xpos[warp_wenv._wenv._ee_site_id]
        
        r_vec = ee_pos - xanchor
        J_cols = np.cross(xaxis, r_vec)
        J = J_cols.T 
        
        JJT = J @ J.T
        det = np.linalg.det(JJT)
        w = np.sqrt(max(0.0, det))
        
        yoshikawa_traj.append(w)
        
    return np.array(yoshikawa_traj)

if __name__ == "__main__":
    base_data_path = r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl"

    print("Evaluating SAC Manipulability...")
    sac_path = os.path.join(base_data_path, "MujocoUR3PushEnv_sac_teste_fps_mesa_20260626_015332")
    sac_model, sac_env, sac_wenv = load_policy(sac_path, "sac")
    sac_yosh = extract_manipulability(sac_model, sac_env, sac_wenv)

    print("Evaluating PPO Manipulability...")
    ppo_path = os.path.join(base_data_path, "MujocoUR3PushEnv_ppo_teste_fps_mesa_20260625_223152")
    ppo_model, ppo_env, ppo_wenv = load_policy(ppo_path, "ppo")
    ppo_yosh = extract_manipulability(ppo_model, ppo_env, ppo_wenv)

    print("Plotting Manipulability...")
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Conforto Cinemático: Índice de Manipulabilidade de Yoshikawa', fontsize=16)

    ax.plot(range(1, 101), sac_yosh, 'r-', label='SAC', linewidth=2, alpha=0.8)
    ax.plot(range(1, 101), ppo_yosh, 'b-', label='PPO', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Passo da Simulação', fontsize=12)
    ax.set_ylabel('Índice w (Yoshikawa)', fontsize=12)
    ax.set_title('Maior é melhor (mais longe de singularidades)', fontsize=14)
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    output_img = os.path.join(os.path.dirname(__file__), "results", "manipulability.png")
    fig.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Saved {output_img}")
