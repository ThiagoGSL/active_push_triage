import os
import numpy as np
import pandas as pd

def analyze_run(run_dir, name):
    print('='*50)
    print(f'Run: {name}')
    prog_file = os.path.join(run_dir, 'logs', 'progress.csv')
    if os.path.exists(prog_file):
        df = pd.read_csv(prog_file)
        if 'time/time_elapsed' in df.columns:
            time_elapsed = df['time/time_elapsed'].iloc[-1]
            hours = time_elapsed / 3600
            print(f'Time Elapsed: {time_elapsed} seconds ({hours:.2f} hours)')
        else:
            print('Time Elapsed: N/A')
            
        if 'time/fps' in df.columns:
            print(f'Final FPS: {df["time/fps"].iloc[-1]}')
        if 'time/total_timesteps' in df.columns:
            print(f'Total Timesteps: {df["time/total_timesteps"].iloc[-1]}')
            
    eval_file = os.path.join(run_dir, 'evaluation', 'evaluations.npz')
    if os.path.exists(eval_file):
        data = np.load(eval_file)
        timesteps = data['timesteps']
        successes = data.get('successes', None)
        if successes is not None:
            print('Success Rate Curve:')
            for i, ts in enumerate(timesteps):
                print(f'  Step {ts}: {np.mean(successes[i]):.2f}')
        else:
            print('No successes array found.')

ppo_dir = r'ur3_push_data\rl\MujocoUR3PushSimpleEnv_warp_1024_envs_cylinder_vecnorm_20260612_015744'
sac_dir = r'ur3_push_data\rl\MujocoUR3PushEnv_cylinder_sac_v4_stable_20260619_012925'

analyze_run(ppo_dir, 'PPO (1024 envs, vecnorm)')
analyze_run(sac_dir, 'SAC (64 envs, v4_stable)')
