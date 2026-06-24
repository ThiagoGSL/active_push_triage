import time
import argparse
import numpy as np
import os
from stable_baselines3.common.vec_env import SubprocVecEnv
from ur3_push_rl_sb3.make_envs import make_vec_envs, make_warp_env

def run_benchmark(env, num_steps=500):
    env.reset()
    start_time = time.time()
    for _ in range(num_steps):
        # random actions
        actions = np.random.uniform(-1, 1, size=(env.num_envs, 2))
        env.step(actions)
    
    total_time = time.time() - start_time
    total_transitions = num_steps * env.num_envs
    fps = total_transitions / total_time
    return fps

def main():
    print("Iniciando benchmark CPU (SubprocVecEnv)...")
    num_envs_cpu = 16
    os.makedirs('dummy_log', exist_ok=True)
    cpu_envs, _ = make_vec_envs(
        "MujocoUR3PushSimpleEnv",
        False,
        'dummy_log',
        SubprocVecEnv,
        num_envs_cpu,
        42,
        42,
        None,
        True,
        max_episode_steps=100,
        use_sim_config=1,
        render_mode="rgb_array",
        n_substeps=40,
        action_scaling_factor=0.02
    )
    
    # Warmup e execucao CPU
    run_benchmark(cpu_envs, 50) # warmup
    fps_cpu = run_benchmark(cpu_envs, 200)
    cpu_envs.close()
    print(f"FPS CPU ({num_envs_cpu} envs): {fps_cpu:.2f}")

    print("\nIniciando benchmark GPU (MuJoCo Warp)...")
    num_envs_gpu = 1024
    gpu_envs, _ = make_warp_env(
        num_train=num_envs_gpu,
        max_episode_steps=100,
        sparse_reward=False,
        n_substeps=40,
        seed=42,
        action_scaling_factor=0.02
    )
    
    # Warmup e execucao GPU
    run_benchmark(gpu_envs, 50) # warmup
    fps_gpu = run_benchmark(gpu_envs, 200)
    gpu_envs.close()
    print(f"FPS GPU Warp ({num_envs_gpu} envs): {fps_gpu:.2f}")

if __name__ == '__main__':
    main()
