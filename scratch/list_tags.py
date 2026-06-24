from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

path = r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl\MujocoUR3PushSimpleEnv_warp_1024_envs_stacked_gold_perfect_20260610_132928\logs"
try:
    ea = EventAccumulator(path)
    ea.Reload()
    print("PPO Tags:", ea.Tags()['scalars'])
except Exception as e:
    print(f"Error: {e}")
