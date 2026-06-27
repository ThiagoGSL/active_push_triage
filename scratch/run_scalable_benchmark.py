import time
import json
import numpy as np
import os
import argparse
from stable_baselines3.common.vec_env import SubprocVecEnv
from ur3_push_rl_sb3.make_envs import make_vec_envs, make_warp_env

def run_benchmark(env, num_steps=200):
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

def benchmark_cpu():
    envs_list = [1, 2, 4, 8, 16, 20, 24]
    results = {}
    print("=== BENCHMARK CPU (SubprocVecEnv) ===")
    os.makedirs('dummy_log', exist_ok=True)
    
    for num_envs in envs_list:
        try:
            print(f"Testando {num_envs} instâncias na CPU...")
            envs, _ = make_vec_envs(
                "MujocoUR3PushSimpleEnv",
                False,
                'dummy_log',
                SubprocVecEnv,
                num_envs,
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
            # warmup
            run_benchmark(envs, 20)
            # benchmark real
            fps = run_benchmark(envs, 100)
            envs.close()
            results[str(num_envs)] = fps
            print(f"  -> FPS: {fps:.2f}")
        except Exception as e:
            print(f"  -> Erro ao rodar com {num_envs} instâncias na CPU: {e}")
            break
    return results

def benchmark_gpu():
    envs_list = [1, 16, 64, 256, 1024, 2048, 4096, 8192]
    results = {}
    print("\n=== BENCHMARK GPU (MuJoCo Warp) ===")
    
    for num_envs in envs_list:
        try:
            print(f"Testando {num_envs} instâncias na GPU Warp...")
            envs, _ = make_warp_env(
                num_train=num_envs,
                max_episode_steps=100,
                sparse_reward=False,
                n_substeps=40,
                seed=42,
                action_scaling_factor=0.02
            )
            # warmup
            run_benchmark(envs, 20)
            # benchmark real
            fps = run_benchmark(envs, 100)
            envs.close()
            results[str(num_envs)] = fps
            print(f"  -> FPS: {fps:.2f}")
        except Exception as e:
            print(f"  -> Erro ou limite de VRAM atingido com {num_envs} instâncias na GPU: {e}")
            break
    return results

def main():
    cpu_results = benchmark_cpu()
    gpu_results = benchmark_gpu()
    
    final_data = {
        "cpu": cpu_results,
        "gpu": gpu_results
    }
    
    with open("scratch/benchmark_results.json", "w") as f:
        json.dump(final_data, f, indent=4)
    print("\nResultados salvos em scratch/benchmark_results.json")

if __name__ == '__main__':
    main()
