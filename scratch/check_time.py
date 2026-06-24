import pandas as pd
import os

ppo_csv = r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl\MujocoUR3PushSimpleEnv_warp_1024_envs_cylinder_vecnorm_20260612_015744\logs\progress.csv"
sac_csv = r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl\MujocoUR3PushEnv_cylinder_sac_v5_final_20260619_091116\logs\progress.csv"

def print_time(name, path):
    print(f"--- {name} ---")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'time/time_elapsed' in df.columns:
            seconds = df['time/time_elapsed'].max()
            minutes = seconds / 60
            hours = minutes / 60
            print(f"Time Elapsed: {seconds:.0f} seconds ({minutes:.1f} min / {hours:.2f} h)")
        else:
            print("time/time_elapsed column not found.")
    else:
        print("CSV not found.")

print_time("PPO", ppo_csv)
print_time("SAC", sac_csv)
