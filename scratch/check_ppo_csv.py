import pandas as pd
csv_path = r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl\MujocoUR3PushSimpleEnv_warp_1024_envs_cylinder_vecnorm_20260612_015744\logs\progress.csv"
df = pd.read_csv(csv_path)
print("Columns:", df.columns.tolist())
if 'time/fps' in df.columns:
    print("Average FPS:", df['time/fps'].mean())
    print("Last FPS:", df['time/fps'].iloc[-1])
if 'time/time_elapsed' in df.columns:
    print("Total Time Elapsed:", df['time/time_elapsed'].max())
