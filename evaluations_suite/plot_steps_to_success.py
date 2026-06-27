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

def extract_steps_to_success(model, eval_env, num_episodes=100):
    steps_list = []
    
    for ep in range(num_episodes):
        obs = eval_env.reset()
        success_step = 100
        for step in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, dones, infos = eval_env.step(action)
            if infos[0].get('is_success', False):
                success_step = step + 1
                break
        steps_list.append(success_step)
        
    return np.array(steps_list)

if __name__ == "__main__":
    base_data_path = r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl"

    print("Evaluating SAC Steps to Success (100 episodes)...")
    sac_path = os.path.join(base_data_path, "MujocoUR3PushEnv_sac_teste_fps_mesa_20260626_015332")
    sac_model, sac_env, sac_wenv = load_policy(sac_path, "sac")
    sac_steps = extract_steps_to_success(sac_model, sac_env, num_episodes=100)

    print("Evaluating PPO Steps to Success (100 episodes)...")
    ppo_path = os.path.join(base_data_path, "MujocoUR3PushEnv_ppo_teste_fps_mesa_20260625_223152")
    ppo_model, ppo_env, ppo_wenv = load_policy(ppo_path, "ppo")
    ppo_steps = extract_steps_to_success(ppo_model, ppo_env, num_episodes=100)

    sac_success_rate = np.mean(sac_steps < 100) * 100
    ppo_success_rate = np.mean(ppo_steps < 100) * 100

    print("Plotting Boxplot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Agilidade de Resolução: Passos até o Sucesso', fontsize=16)

    data = [sac_steps[sac_steps < 100], ppo_steps[ppo_steps < 100]]
    if len(data[0]) == 0: data[0] = [100]
    if len(data[1]) == 0: data[1] = [100]

    box = ax.boxplot(data, patch_artist=True, labels=[f'SAC\n(Sucesso: {sac_success_rate:.0f}%)', f'PPO\n(Sucesso: {ppo_success_rate:.0f}%)'])

    colors = ['#ff9999', '#9999ff']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Quantidade de Passos (Steps)', fontsize=12)
    ax.set_title('Apenas episódios com sucesso (menor é mais rápido)', fontsize=14)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    output_img = os.path.join(os.path.dirname(__file__), "results", "steps_to_success.png")
    fig.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Saved {output_img}")
