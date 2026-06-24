from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ppo_path = r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl\MujocoUR3PushSimpleEnv_warp_1024_envs_cylinder_vecnorm_20260612_015744\logs"
sac_path = r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl\MujocoUR3PushEnv_cylinder_sac_v5_final_20260619_091116\logs"

def check_time(name, path):
    ea = EventAccumulator(path)
    ea.Reload()
    if 'eval/success_rate' in ea.Tags()['scalars']:
        events = ea.Scalars('eval/success_rate')
        if events:
            first_time = events[0].wall_time
            last_time = events[-1].wall_time
            print(f"--- {name} ---")
            print(f"First Eval Time: {first_time}")
            print(f"Last Eval Time: {last_time}")
            print(f"Total Elapsed for Eval: {last_time - first_time} seconds ({(last_time - first_time)/3600:.2f} hours)")
        
            # check the very first logged event to get true start time
            first_overall = min([ea.Scalars(tag)[0].wall_time for tag in ea.Tags()['scalars'] if ea.Scalars(tag)])
            print(f"True Start Time: {first_overall}")
            print(f"Total Elapsed from True Start: {last_time - first_overall} seconds ({(last_time - first_overall)/3600:.2f} hours)")

check_time("PPO", ppo_path)
check_time("SAC", sac_path)
