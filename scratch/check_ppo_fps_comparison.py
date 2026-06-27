import pandas as pd
import os

old_csv = r'ur3_push_data\rl\MujocoUR3PushEnv_sac_definitivo_10M_Warp_GPU_20260625_062752\logs\progress.csv'
new_csv = r'ur3_push_data\rl\MujocoUR3PushEnv_sac_teste_fps_mesa_20260626_015332\logs\progress.csv'

if os.path.exists(old_csv):
    df_old = pd.read_csv(old_csv)
    if 'time/fps' in df_old.columns:
        fps_col = df_old['time/fps']
        print(f'Old FPS Avg: {fps_col.iloc[2:].mean():.2f}')
if os.path.exists(new_csv):
    df_new = pd.read_csv(new_csv)
    if 'time/fps' in df_new.columns:
        fps_col = df_new['time/fps']
        print(f'New FPS Avg: {fps_col.iloc[2:].mean():.2f}')
