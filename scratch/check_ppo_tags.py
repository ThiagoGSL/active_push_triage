from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

ppo_path = r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl\MujocoUR3PushSimpleEnv_warp_1024_envs_cylinder_vecnorm_20260612_015744\logs"
ea = EventAccumulator(ppo_path)
ea.Reload()

print("Tags:", ea.Tags()['scalars'])
for tag in ea.Tags()['scalars']:
    events = ea.Scalars(tag)
    print(f"{tag}: length {len(events)}, first step {events[0].step}")
