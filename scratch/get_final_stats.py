import pandas as pd
import os

ppo_path = r"ur3_push_data\rl\MujocoUR3PushEnv_ppo_definitivo_30M_STACKED_20260625_013827\logs\progress.csv"
sac_path = r"ur3_push_data\rl\MujocoUR3PushEnv_sac_definitivo_10M_Warp_GPU_20260625_062752\logs\progress.csv"

def print_stats(name, path):
    df = pd.read_csv(path)
    fps_col = 'time/fps'
    time_col = 'time/time_elapsed'
    
    mean_fps = df[fps_col].mean()
    total_time_s = df[time_col].iloc[-1]
    total_time_h = total_time_s / 3600
    
    print(f"--- {name} ---")
    print(f"Mean FPS: {mean_fps:.0f}")
    print(f"Total Time: {total_time_h:.2f} hours ({total_time_s:.0f} seconds)")
    print()

print_stats("PPO", ppo_path)
print_stats("SAC", sac_path)
