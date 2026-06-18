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
        seed=train_config.get("eEvalSeed", 42),
        torque_penalty_scale=train_config.get("torquePenaltyScale", 0.001),
        action_rate_penalty_scale=train_config.get("actionRatePenaltyScale", 0.05),
        success_bonus=train_config.get("successBonus", 10.0),
        early_termination_on_success=False, # Forçado para False para gravar o pós-sucesso
        randomize_initial_joints=bool(train_config.get("randomizeInitialJoints", 1)),
        manipulability_reward_scale=train_config.get("manipulabilityRewardScale", 0.01),
        manipulability_metric=train_config.get("manipulabilityMetric", "yoshikawa")
    )
    
    warp_wenv = eval_env
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

    obs = eval_env.reset()

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
        
        # Sync GPU to CPU for rendering
        d.qpos[:] = warp_wenv._wenv._d_gpu.qpos.numpy()[0]
        d.qvel[:] = warp_wenv._wenv._d_gpu.qvel.numpy()[0]
        
        # Update target body position visually
        try:
            target_xy = warp_wenv._wenv._target_xypos[0]
            body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target')
            if body_id >= 0:
                m.body_pos[body_id][0] = target_xy[0]
                m.body_pos[body_id][1] = target_xy[1]
        except Exception as e:
            pass
        
        # Site positions are important to update object visual positions if they are mocap or kinematic
        if hasattr(warp_wenv._wenv._d_gpu, 'mocap_pos'):
            try:
                d.mocap_pos[:] = warp_wenv._wenv._d_gpu.mocap_pos.numpy()[0]
                d.mocap_quat[:] = warp_wenv._wenv._d_gpu.mocap_quat.numpy()[0]
            except: pass
            
        mujoco.mj_forward(m, d)
        
        # Render frame
        renderer.update_scene(d, camera=-1)
        pixels = renderer.render()
        writer.append_data(pixels)
        
        if dones[0]:
            is_success = infos[0].get('is_success', False)
            successes.append(is_success)
            print(f"Episode {episodes}: Success = {is_success}")
            episodes += 1
            steps_in_episode = 0

    writer.close()
    
    success_rate = np.mean(successes) * 100
    print(f"\nFinal Success Rate: {success_rate:.2f}% ({sum(successes)}/{args.eNumEvalEpisodes})")
    print(f"Video saved successfully at: {video_path}")

if __name__ == "__main__":
    main()
