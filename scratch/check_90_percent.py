from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ppo_path = r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl\MujocoUR3PushSimpleEnv_warp_1024_envs_cylinder_vecnorm_20260612_015744\logs"
sac_path = r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl\MujocoUR3PushEnv_cylinder_sac_v5_final_20260619_091116\logs"

def check_90(path):
    ea = EventAccumulator(path)
    ea.Reload()
    if 'eval/success_rate' in ea.Tags()['scalars']:
        events = ea.Scalars('eval/success_rate')
        for e in events:
            if e.value >= 0.9:
                return e.step
    return -1

print("PPO 90% at step:", check_90(ppo_path))
print("SAC 90% at step:", check_90(sac_path))
