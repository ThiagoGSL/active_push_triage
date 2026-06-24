import pandas as pd
import os

ppo_csv = r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl\MujocoUR3PushSimpleEnv_warp_1024_envs_stacked_gold_perfect_20260610_132928\logs\progress.csv"
sac_csv = r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl\MujocoUR3PushEnv_cylinder_sac_v5_final_20260619_091116\logs\progress.csv"

def print_stats(name, path):
    print(f"--- {name} ---")
    if os.path.exists(path):
        df = pd.read_csv(path)
        print("Columns:", df.columns.tolist()[:10])
        if 'eval/success_rate' in df.columns:
            print(f"Max Success Rate: {df['eval/success_rate'].max():.4f}")
            print(f"Last Success Rate: {df['eval/success_rate'].iloc[-1]:.4f}")
        if 'eval/mean_reward' in df.columns:
            print(f"Max Reward: {df['eval/mean_reward'].max():.4f}")
        if 'time/total_timesteps' in df.columns:
            print(f"Total steps: {df['time/total_timesteps'].max()}")
        elif 'time/total timesteps' in df.columns:
            print(f"Total steps: {df['time/total timesteps'].max()}")
    else:
        print("File not found.")

print_stats("PPO", ppo_csv)
print_stats("SAC", sac_csv)
