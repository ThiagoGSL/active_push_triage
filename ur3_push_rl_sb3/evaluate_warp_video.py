import os
import argparse
import json
import numpy as np
import imageio
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecFrameStack
import mujoco

from ur3_push_rl_sb3.make_envs import make_warp_env

def _load_model_by_algorithm(config_path: str, model_path: str, env):
    """Carrega PPO ou SAC detectando automaticamente o algoritmo do config.txt.
    Runs antigas sem o campo 'algorithm' defaultam para PPO (retrocompatibilidade).
    """
    algorithm = "ppo"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            algorithm = json.load(f).get("algorithm", "ppo")
    algorithm = algorithm.lower()
    print(f"[Evaluate] Detected algorithm: {algorithm.upper()}")
    if algorithm == "sac":
        return SAC.load(model_path, env=env)
    return PPO.load(model_path, env=env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evalPath", type=str, required=True, help="Path to the training directory")
    parser.add_argument("--eVideoName", type=str, default="warp_eval_video", help="Name of the output video")
    parser.add_argument("--numStackedObs", type=int, default=4, help="Number of stacked observations")
    parser.add_argument("--eNumEvalEpisodes", type=int, default=5, help="Number of episodes to record")
    parser.add_argument("--dynamicTarget", type=str, choices=["none", "circle", "rectangle"], default="none", help="Move the target dynamically ('circle' or 'rectangle')")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(args.evalPath, "logs", "config.txt")
    with open(config_path, "r") as f:
        train_config = json.load(f)

    # Initialize Warp Environment
    _, eval_env = make_warp_env(
        num_train=1,
        max_episode_steps=train_config.get("maxEpisodeSteps", 100),
        sparse_reward=bool(train_config.get("sparseReward", 0)),
        ee_to_obj_reward_scale=0.0,
        action_scaling_factor=train_config.get("actionScalingFactor", 0.05),
        n_substeps=train_config.get("numSimSteps", 120),
        seed=np.random.randint(0, 100000), # Garante posições aleatórias a cada execução
        torque_penalty_scale=train_config.get("torquePenaltyScale", 0.001),
        action_rate_penalty_scale=train_config.get("actionRatePenaltyScale", 0.05),
        success_bonus=train_config.get("successBonus", 0.0),
        early_termination_on_success=0, # FORÇADO: Não terminar para podermos gravar o robô parado
        randomize_initial_joints=bool(train_config.get("randomizeInitialJoints", 0)),
        manipulability_reward_scale=train_config.get("manipulabilityRewardScale", 0.01),
        manipulability_metric=train_config.get("manipulabilityMetric", "yoshikawa")
    )
    
    warp_wenv = eval_env
    if args.numStackedObs > 1:
        eval_env = VecFrameStack(eval_env, n_stack=args.numStackedObs)

    model_path = os.path.join(args.evalPath, "evaluation", "best_model")
    if not os.path.exists(model_path + ".zip"):
        model_path = os.path.join(args.evalPath, "best_model")
    
    config_path = os.path.join(args.evalPath, "logs", "config.txt")
    model = _load_model_by_algorithm(config_path, model_path, eval_env)

    # Prepare renderer
    m = warp_wenv._wenv._mjm
    d = warp_wenv._wenv._cpu_data
    renderer = mujoco.Renderer(m, 480, 640)

    videos_dir = os.path.join(os.path.dirname(args.evalPath), "videos")
    os.makedirs(videos_dir, exist_ok=True)
    video_path = os.path.join(videos_dir, f"{args.eVideoName}.mp4")
    
    writer = imageio.get_writer(video_path, fps=30)
    print(f"Recording video to {video_path}...")

    successes = []
    episodes = 0
    steps_in_episode = 0
    
    if args.dynamicTarget != "none":
        # Força o objeto a surgir sempre no centro geométrico da mesa
        warp_wenv._wenv._sample_xy_away_from = lambda *args, **kwargs: np.array([0.375, 0.0])
    else:
        # Posições pre-programadas para garantir benchmark justo e comparável
        preprogrammed_positions = [
            (np.array([0.30, -0.15]), np.array([0.45,  0.15])), # 1. Diagonal Longa (inferior esquerdo para superior direito)
            (np.array([0.45, 0.15]), np.array([0.45,  -0.15])), # 2. Linha Reta Y Oposta (superior direito para superior esquerdo)
            (np.array([0.45,  -0.15]), np.array([0.30, 0.15])), # 3. Diagonal Longa (superior esquerdo para inferior direito)
            (np.array([0.45,  0.15]), np.array([0.30, -0.15])), # 4. Diagonal Longa Oposta (superior direito para inferior esquerdo)
            (np.array([0.375, -0.15]), np.array([0.375, 0.15])), # 5. Linha Reta Y (esquerda para direita)
            (np.array([0.375, 0.15]), np.array([0.375, -0.15])), # 6. Linha Reta Y (direita para esquerda)
            (np.array([0.30, -0.15]), np.array([0.45, -0.15])),  # 7. Linha Reta X (canto esquerdo)
            (np.array([0.30, 0.0]), np.array([0.45, 0.0])),      # 8. Linha Reta X (centro)
            #(np.array([0.30, 0.15]), np.array([0.45, 0.15])),    # 9. Linha Reta X (canto direito)
            #(np.array([0.32, 0.0]), np.array([0.45, 0.05])),     # 10. Empurrão Curto/Desalinhado
        ]

        class PositionSampler:
            def __init__(self):
                self.call_count = 0
                
            def __call__(self, ref_xy, min_dist):
                episode_idx = (self.call_count // 2) % len(preprogrammed_positions)
                is_target = (self.call_count % 2) == 1
                self.call_count += 1
                if is_target:
                    return preprogrammed_positions[episode_idx][1]
                else:
                    return preprogrammed_positions[episode_idx][0]

        warp_wenv._wenv._sample_xy_away_from = PositionSampler()

    obs = eval_env.reset()
    has_succeeded_this_episode = False

    video_time = 0.0
    physics_time = 0.0
    fps = 30
    dt_video = 1.0 / fps
    dt_physics = train_config.get("numSimSteps", 120) * 0.001

    # Grab initial state
    prev_qpos = warp_wenv._wenv._d_gpu.qpos.numpy()[0].copy()
    
    if hasattr(warp_wenv._wenv._d_gpu, 'mocap_pos'):
        prev_mocap_pos = warp_wenv._wenv._d_gpu.mocap_pos.numpy()[0].copy()
        prev_mocap_quat = warp_wenv._wenv._d_gpu.mocap_quat.numpy()[0].copy()
    else:
        prev_mocap_pos = None
        prev_mocap_quat = None

    while episodes < args.eNumEvalEpisodes:
        if args.dynamicTarget == "circle":
            t = steps_in_episode * 0.025  # velocidade angular (reduzida pela metade)
            warp_wenv._wenv._target_xypos[0, 0] = 0.375 + 0.08 * np.cos(t) # raio reduzido para 8cm
            warp_wenv._wenv._target_xypos[0, 1] = 0.0 + 0.08 * np.sin(t)
        elif args.dynamicTarget == "rectangle":
            # Perímetro paramétrico (0 a 4)
            t = (steps_in_episode * 0.0075) % 4 # velocidade linear (reduzida pela metade)
            if t < 1:   # Aresta superior (esq -> dir)
                dx, dy = -0.08 + (t - 0) * 0.16,  0.06
            elif t < 2: # Aresta direita (cima -> baixo)
                dx, dy =  0.08,  0.06 - (t - 1) * 0.12
            elif t < 3: # Aresta inferior (dir -> esq)
                dx, dy =  0.08 - (t - 2) * 0.16, -0.06
            else:       # Aresta esquerda (baixo -> cima)
                dx, dy = -0.08, -0.06 + (t - 3) * 0.12
            
            warp_wenv._wenv._target_xypos[0, 0] = 0.375 + dx
            warp_wenv._wenv._target_xypos[0, 1] = 0.0 + dy
            
        action, _ = model.predict(obs, deterministic=True)
        obs, r, dones, infos = eval_env.step(action)
        steps_in_episode += 1
        
        # Trava o status de sucesso para o resto do episódio
        if infos[0].get('is_success', False):
            has_succeeded_this_episode = True
        
        # Save current state for interpolation
        curr_qpos = warp_wenv._wenv._d_gpu.qpos.numpy()[0].copy()
        if prev_mocap_pos is not None:
            curr_mocap_pos = warp_wenv._wenv._d_gpu.mocap_pos.numpy()[0].copy()
            curr_mocap_quat = warp_wenv._wenv._d_gpu.mocap_quat.numpy()[0].copy()
            
        physics_time += dt_physics
        
        while video_time <= physics_time:
            alpha = (video_time - (physics_time - dt_physics)) / dt_physics
            alpha = max(0.0, min(1.0, alpha))
            
            d.qpos[:] = prev_qpos + alpha * (curr_qpos - prev_qpos)
            
            # Update target body position visually
            try:
                target_xy = warp_wenv._wenv._target_xypos[0]
                body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target')
                if body_id >= 0:
                    m.body_pos[body_id][0] = target_xy[0]
                    m.body_pos[body_id][1] = target_xy[1]
            except Exception as e:
                pass
            
            if prev_mocap_pos is not None:
                try:
                    d.mocap_pos[:] = prev_mocap_pos + alpha * (curr_mocap_pos - prev_mocap_pos)
                    d.mocap_quat[:] = prev_mocap_quat + alpha * (curr_mocap_quat - prev_mocap_quat)
                except: pass
                
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera="eval_camera")
            pixels = renderer.render()
            writer.append_data(pixels)
            
            video_time += dt_video
            
        # Shift state
        prev_qpos = curr_qpos.copy()
        if prev_mocap_pos is not None:
            prev_mocap_pos = curr_mocap_pos.copy()
            prev_mocap_quat = curr_mocap_quat.copy()
        
        if dones[0]:
            successes.append(has_succeeded_this_episode)
            print(f"Episode {episodes}: Success = {has_succeeded_this_episode}")
            episodes += 1
            steps_in_episode = 0
            has_succeeded_this_episode = False

    writer.close()
    
    success_rate = np.mean(successes) * 100
    print(f"\nFinal Success Rate: {success_rate:.2f}% ({sum(successes)}/{args.eNumEvalEpisodes})")
    print(f"Video saved successfully at: {video_path}")

if __name__ == "__main__":
    main()
