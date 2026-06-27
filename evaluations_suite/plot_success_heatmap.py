import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

def extract_heatmap(model, eval_env, warp_wenv, x_range, y_range):
    class BetterGridSampler:
        def __init__(self, obj_pos, target_pos):
            self.obj_pos = obj_pos
            self.target_pos = target_pos
            self.call_count = 0
        def __call__(self, ref_xy, min_dist):
            is_target = (self.call_count % 2) == 1
            self.call_count += 1
            if is_target:
                return self.target_pos
            return self.obj_pos

    success_matrix = np.zeros((len(y_range), len(x_range)))
    target_pos = np.array([0.45, 0.0])
    
    for i, y in enumerate(y_range):
        for j, x in enumerate(x_range):
            test_pos = np.array([x, y])
            warp_wenv._wenv._sample_xy_away_from = BetterGridSampler(test_pos, target_pos)
            
            obs = eval_env.reset()
            success = False
            for _ in range(100):
                action, _ = model.predict(obs, deterministic=True)
                obs, r, dones, infos = eval_env.step(action)
                if infos[0].get('is_success', False):
                    success = True
                    break
            success_matrix[i, j] = 1.0 if success else 0.0
            
    return success_matrix

if __name__ == "__main__":
    base_data_path = r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl"
    
    # Define grid 10x10
    x_range = np.linspace(0.20, 0.50, 10)
    y_range = np.linspace(-0.25, 0.25, 10)

    print("Evaluating SAC Heatmap...")
    sac_path = os.path.join(base_data_path, "MujocoUR3PushEnv_sac_teste_fps_mesa_20260626_015332")
    sac_model, sac_env, sac_wenv = load_policy(sac_path, "sac")
    sac_heatmap = extract_heatmap(sac_model, sac_env, sac_wenv, x_range, y_range)

    print("Evaluating PPO Heatmap...")
    ppo_path = os.path.join(base_data_path, "MujocoUR3PushEnv_ppo_teste_fps_mesa_20260625_223152")
    ppo_model, ppo_env, ppo_wenv = load_policy(ppo_path, "ppo")
    ppo_heatmap = extract_heatmap(ppo_model, ppo_env, ppo_wenv, x_range, y_range)

    print("Plotting Heatmap...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Mapa Espacial de Sucesso: Alvo Fixo (X=0.45, Y=0.0)', fontsize=16)

    cmap = mcolors.ListedColormap(['red', 'green'])
    
    # Extent for plotting correctly: [xmin, xmax, ymin, ymax]
    extent = [x_range[0], x_range[-1], y_range[0], y_range[-1]]
    
    im1 = ax1.imshow(sac_heatmap, cmap=cmap, origin='lower', extent=extent, alpha=0.6)
    ax1.scatter(0.45, 0.0, color='gold', marker='*', s=200, edgecolors='black', label='Alvo')
    ax1.set_title('SAC: Zonas de Sucesso')
    ax1.set_xlabel('Posição X Inicial (m)')
    ax1.set_ylabel('Posição Y Inicial (m)')
    ax1.legend()
    ax1.grid(True, color='white', linewidth=0.5)

    im2 = ax2.imshow(ppo_heatmap, cmap=cmap, origin='lower', extent=extent, alpha=0.6)
    ax2.scatter(0.45, 0.0, color='gold', marker='*', s=200, edgecolors='black', label='Alvo')
    ax2.set_title('PPO: Zonas de Sucesso')
    ax2.set_xlabel('Posição X Inicial (m)')
    ax2.set_ylabel('Posição Y Inicial (m)')
    ax2.legend()
    ax2.grid(True, color='white', linewidth=0.5)

    fig.tight_layout()
    output_img = os.path.join(os.path.dirname(__file__), "results", "success_heatmap.png")
    fig.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Saved {output_img}")
