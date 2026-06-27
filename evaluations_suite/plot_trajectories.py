import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

def extract_trajectory(model, eval_env, warp_wenv):
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
    ee_traj = []
    obj_traj = []
    torque_traj = []
    
    ee_pos = warp_wenv._wenv._d_gpu.site_xpos.numpy()[0, warp_wenv._wenv._ee_site_id, :2].copy()
    obj_pos = warp_wenv._wenv._d_gpu.site_xpos.numpy()[0, warp_wenv._wenv._obj_site_id, :2].copy()
    target_pos = warp_wenv._wenv._target_xypos[0].copy()
    
    ee_traj.append(ee_pos)
    obj_traj.append(obj_pos)
    torque_traj.append(0.0)
    
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, dones, infos = eval_env.step(action)
        
        ee_pos = warp_wenv._wenv._d_gpu.site_xpos.numpy()[0, warp_wenv._wenv._ee_site_id, :2].copy()
        obj_pos = warp_wenv._wenv._d_gpu.site_xpos.numpy()[0, warp_wenv._wenv._obj_site_id, :2].copy()
        
        qfrc = warp_wenv._wenv._d_gpu.qfrc_actuator.numpy()[0, :6]
        torque_sum = np.sum(np.abs(qfrc))
        
        ee_traj.append(ee_pos)
        obj_traj.append(obj_pos)
        torque_traj.append(torque_sum)
        
        if infos[0].get('is_success', False):
            pass
            
    final_dist = np.linalg.norm(obj_pos - target_pos)
    return np.array(ee_traj), np.array(obj_traj), target_pos, np.array(torque_traj), final_dist

base_data_path = r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl"

print("Evaluating SAC...")
sac_path = os.path.join(base_data_path, "MujocoUR3PushEnv_sac_teste_fps_mesa_20260626_015332")
sac_model, sac_env, sac_wenv = load_policy(sac_path, "sac")
sac_ee, sac_obj, target_pos, sac_torque, sac_dist = extract_trajectory(sac_model, sac_env, sac_wenv)

print("Evaluating PPO...")
ppo_path = os.path.join(base_data_path, "MujocoUR3PushEnv_ppo_teste_fps_mesa_20260625_223152")
ppo_model, ppo_env, ppo_wenv = load_policy(ppo_path, "ppo")
ppo_ee, ppo_obj, _, ppo_torque, ppo_dist = extract_trajectory(ppo_model, ppo_env, ppo_wenv)

print("Plotting...")
print("Plotting...")

# --- FIGURE 1: Trajectories ---
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
fig1.suptitle('Análise Comparativa: PPO vs SAC', fontsize=18)

# --- SAC Trajectory ---
ax1.plot(sac_ee[:, 0], sac_ee[:, 1], 'r--', label='SAC End-Effector', alpha=0.7)
ax1.plot(sac_obj[:, 0], sac_obj[:, 1], 'r-', label='SAC Objeto (Centro)', linewidth=1, alpha=0.5)

num_steps_sac = len(sac_obj)
for i, pos in enumerate(sac_obj):
    alpha_val = 0.01 + 0.14 * (i / max(1, num_steps_sac - 1))
    circle = patches.Circle(pos, radius=0.055, color='red', alpha=alpha_val, lw=0)
    ax1.add_patch(circle)

ax1.scatter(target_pos[0], target_pos[1], color='gold', marker='*', s=400, label='Alvo', zorder=10, edgecolors='black')
ax1.scatter(sac_obj[0, 0], sac_obj[0, 1], color='black', marker='X', s=150, label='Posição Inicial', zorder=10)

ax1.set_xlim([0.2, 0.55])
ax1.set_ylim([-0.25, 0.25])
ax1.set_xlabel('X (m)', fontsize=12)
ax1.set_ylabel('Y (m)', fontsize=12)
ax1.set_title(f'SAC (Distância Final: {sac_dist:.4f} m)', fontsize=14)
ax1.legend(loc='upper left')
ax1.grid(True)
ax1.set_aspect('equal')

# --- PPO Trajectory ---
ax2.plot(ppo_ee[:, 0], ppo_ee[:, 1], 'b--', label='PPO End-Effector', alpha=0.7)
ax2.plot(ppo_obj[:, 0], ppo_obj[:, 1], 'b-', label='PPO Objeto (Centro)', linewidth=1, alpha=0.5)

num_steps_ppo = len(ppo_obj)
for i, pos in enumerate(ppo_obj):
    alpha_val = 0.01 + 0.14 * (i / max(1, num_steps_ppo - 1))
    circle = patches.Circle(pos, radius=0.055, color='blue', alpha=alpha_val, lw=0)
    ax2.add_patch(circle)

ax2.scatter(target_pos[0], target_pos[1], color='gold', marker='*', s=400, label='Alvo', zorder=10, edgecolors='black')
ax2.scatter(ppo_obj[0, 0], ppo_obj[0, 1], color='black', marker='X', s=150, label='Posição Inicial', zorder=10)

ax2.set_xlim([0.2, 0.55])
ax2.set_ylim([-0.25, 0.25])
ax2.set_xlabel('X (m)', fontsize=12)
ax2.set_title(f'PPO (Distância Final: {ppo_dist:.4f} m)', fontsize=14)
ax2.legend(loc='upper left')
ax2.grid(True)
ax2.set_aspect('equal')

fig1.tight_layout()
output_img_traj = os.path.join(r"c:\Projetos\TCC\active_push_triage\evaluations_suite\results", "trajectory_comparison.png")
fig1.savefig(output_img_traj, dpi=300, bbox_inches='tight')
print(f"Saved {output_img_traj}")

# --- FIGURE 2: Torque Curve ---
fig2, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(range(num_steps_sac), sac_torque, 'r-', label='SAC', linewidth=2, alpha=0.8)
ax3.plot(range(num_steps_ppo), ppo_torque, 'b-', label='PPO', linewidth=2, alpha=0.8)
ax3.set_xlabel('Passo da Simulação', fontsize=12)
ax3.set_ylabel('Soma Absoluta dos Torques Articulares (Nm)', fontsize=12)
ax3.set_title('Esforço Articular ao Longo do Episódio', fontsize=14)
ax3.legend()
ax3.grid(True)

fig2.tight_layout()
output_img_torque = os.path.join(r"c:\Projetos\TCC\active_push_triage\evaluations_suite\results", "torque_comparison.png")
fig2.savefig(output_img_torque, dpi=300, bbox_inches='tight')
print(f"Saved {output_img_torque}")
