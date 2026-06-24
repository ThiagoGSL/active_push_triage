from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ppo_path = r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl\MujocoUR3PushSimpleEnv_warp_1024_envs_cylinder_vecnorm_20260612_015744\logs"

ea = EventAccumulator(ppo_path)
ea.Reload()

print("Tags:", ea.Tags()['scalars'])

if 'time/time_elapsed' in ea.Tags()['scalars']:
    events = ea.Scalars('time/time_elapsed')
    if events:
        s = events[-1].value
        print(f"PPO Time Elapsed: {s} seconds ({s/60:.1f} min)")
elif 'time/fps' in ea.Tags()['scalars']:
    events = ea.Scalars('time/fps')
    if events:
        avg_fps = sum(e.value for e in events) / len(events)
        print(f"PPO Avg FPS: {avg_fps}")
        print(f"Estimated time for 28M steps: {28000000 / avg_fps} seconds")

