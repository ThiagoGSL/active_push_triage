from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

paths = {
    "PPO (Warp)": r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl\MujocoUR3PushSimpleEnv_warp_1024_envs_stacked_gold_perfect_20260610_132928\logs",
    "SAC (Warp)": r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl\MujocoUR3PushEnv_cylinder_sac_v5_final_20260619_091116\logs"
}

for name, path in paths.items():
    try:
        ea = EventAccumulator(path)
        ea.Reload()
        if 'time/fps' in ea.Tags()['scalars']:
            fps_events = ea.Scalars('time/fps')
            fps_values = [e.value for e in fps_events]
            print(f"{name} -> Mean FPS: {sum(fps_values)/len(fps_values):.2f}, Max FPS: {max(fps_values):.2f}")
        else:
            print(f"{name} -> 'time/fps' not found in tags.")
    except Exception as e:
        print(f"Error reading {name}: {e}")
