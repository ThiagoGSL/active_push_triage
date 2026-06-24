from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sac_path = r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl\MujocoUR3PushEnv_cylinder_sac_v1_20M_20260618_215858\logs"
ea = EventAccumulator(sac_path)
ea.Reload()

print("Tags:", ea.Tags()['scalars'])
if 'rollout/success_rate' in ea.Tags()['scalars']:
    events = ea.Scalars('rollout/success_rate')
    print(f"SAC 20M: length {len(events)}, first step {events[0].step}, last step {events[-1].step}")
    
    first_time = events[0].wall_time
    last_time = events[-1].wall_time
    print(f"Total time: {(last_time - first_time)/3600:.2f} hours")
