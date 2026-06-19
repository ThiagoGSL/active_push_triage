import os
import numpy as np
import pandas as pd

run_dir = r'ur3_push_data\rl\MujocoUR3PushEnv_cylinder_sac_v5_final_20260619_091116'
prog_file = os.path.join(run_dir, 'logs', 'progress.csv')
eval_file = os.path.join(run_dir, 'evaluation', 'evaluations.npz')

if os.path.exists(prog_file):
    df = pd.read_csv(prog_file)
    last_row = df.iloc[-1]
    print('--- TRAINING PROGRESS ---')
    print(f'Total Timesteps: {last_row.get("time/total_timesteps", "N/A")}')
    print(f'Actor Loss: {last_row.get("train/actor_loss", "N/A")}')
    print(f'Critic Loss: {last_row.get("train/critic_loss", "N/A")}')
    print(f'Ent Coef: {last_row.get("train/ent_coef", "N/A")}')
    print(f'FPS: {last_row.get("time/fps", "N/A")}')
else:
    print('No progress.csv found')

if os.path.exists(eval_file):
    data = np.load(eval_file)
    timesteps = data['timesteps']
    results = data['results']
    if 'successes' in data:
        successes = data['successes']
    else:
        successes = None
    print('\n--- EVALUATION RESULTS ---')
    for i in range(len(timesteps)):
        mean_rew = np.mean(results[i])
        mean_succ = np.mean(successes[i]) if successes is not None else 'N/A'
        print(f'Step: {timesteps[i]} | Mean Reward: {mean_rew:.2f} | Success Rate: {mean_succ}')
else:
    print('No evaluations.npz found')
