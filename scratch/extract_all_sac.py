import os, glob, numpy as np, pandas as pd

base_dir = r'ur3_push_data\rl'
sac_dirs = glob.glob(os.path.join(base_dir, '*sac*'))

for d in sorted(sac_dirs):
    print('='*50)
    print(f'Run: {os.path.basename(d)}')
    
    prog_file = os.path.join(d, 'logs', 'progress.csv')
    if os.path.exists(prog_file):
        try:
            df = pd.read_csv(prog_file)
            last_row = df.iloc[-1]
            print(f'Total Timesteps: {last_row.get("time/total_timesteps", "N/A")}')
            print(f'Critic Loss: {last_row.get("train/critic_loss", "N/A")}')
            print(f'Ent Coef: {last_row.get("train/ent_coef", "N/A")}')
            print(f'FPS: {last_row.get("time/fps", "N/A")}')
        except Exception as e:
            print(f'Error reading progress: {e}')
            
    eval_file = os.path.join(d, 'evaluation', 'evaluations.npz')
    if os.path.exists(eval_file):
        try:
            data = np.load(eval_file)
            timesteps = data['timesteps']
            results = data['results']
            successes = data.get('successes', None)
            if len(timesteps) > 0:
                print(f'Evaluations: {len(timesteps)}')
                print(f'Last Eval Step: {timesteps[-1]} | Mean Rew: {np.mean(results[-1]):.2f} | Succ: {np.mean(successes[-1]) if successes is not None else "N/A"}')
        except Exception as e:
            print(f'Error reading evals: {e}')
