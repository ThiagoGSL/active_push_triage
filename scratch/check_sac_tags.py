from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

sac_path = r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl\MujocoUR3PushEnv_cylinder_sac_v5_final_20260619_091116\logs"
ea = EventAccumulator(sac_path)
ea.Reload()

print("Tags:", ea.Tags()['scalars'])
for tag in ea.Tags()['scalars']:
    events = ea.Scalars(tag)
    print(f"{tag}: length {len(events)}, first step {events[0].step}")
