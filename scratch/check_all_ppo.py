import pandas as pd
import os
import glob

base_dir = r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl"
all_dirs = glob.glob(os.path.join(base_dir, "MujocoUR3PushSimpleEnv_warp_1024_envs_*"))

results = []
for d in all_dirs:
    csv_path = os.path.join(d, "logs", "progress.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if 'eval/success_rate' in df.columns:
                max_succ = df['eval/success_rate'].max()
                max_rew = df['eval/mean_reward'].max() if 'eval/mean_reward' in df.columns else None
                steps = df['time/total_timesteps'].max() if 'time/total_timesteps' in df.columns else None
                results.append((os.path.basename(d), max_succ, max_rew, steps))
        except Exception as e:
            pass

results.sort(key=lambda x: x[1], reverse=True)
print("Top 5 PPO runs by Success Rate:")
for r in results[:5]:
    print(f"{r[0]}: Succ={r[1]:.4f}, Rew={r[2]}, Steps={r[3]}")
