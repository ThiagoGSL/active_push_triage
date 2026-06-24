import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import os

ppo_path = r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl\MujocoUR3PushSimpleEnv_warp_1024_envs_cylinder_vecnorm_20260612_015744\logs"
sac_path = r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl\MujocoUR3PushEnv_cylinder_sac_v5_final_20260619_091116\logs"

def get_time_data(path, tag):
    ea = EventAccumulator(path)
    ea.Reload()
    if tag not in ea.Tags()['scalars']:
        if tag == 'eval/success_rate' and 'rollout/success_rate' in ea.Tags()['scalars']:
            events = ea.Scalars('rollout/success_rate')
        else:
            return np.array([]), np.array([])
    else:
        events = ea.Scalars(tag)
        if 'sac' in path.lower() and tag == 'eval/success_rate' and 'rollout/success_rate' in ea.Tags()['scalars']:
             events = ea.Scalars('rollout/success_rate')
    
    # Try to find the true start time from any scalar
    try:
        start_time = min([ea.Scalars(t)[0].wall_time for t in ea.Tags()['scalars'] if ea.Scalars(t)])
    except:
        start_time = events[0].wall_time
        
    times_h = [(e.wall_time - start_time)/3600.0 for e in events]
    vals = [e.value for e in events]
    
    if len(times_h) > 0 and times_h[0] > 0:
        times_h.insert(0, 0.0)
        if 'success' in tag.lower():
            vals.insert(0, 0.0)
        else:
            vals.insert(0, vals[0])

    return np.array(times_h), np.array(vals)

def smooth(scalars, weight):
    if len(scalars) == 0: return scalars
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

print("Extraindo dados de tempo...")
ppo_time_r, ppo_rew = get_time_data(ppo_path, 'eval/mean_reward')
sac_time_r, sac_rew = get_time_data(sac_path, 'eval/mean_reward')
ppo_time_s, ppo_succ = get_time_data(ppo_path, 'eval/success_rate')
sac_time_s, sac_succ = get_time_data(sac_path, 'eval/success_rate')

SMOOTH = 0.25
ppo_rew_s = smooth(ppo_rew, SMOOTH) if len(ppo_rew) > 0 else []
sac_rew_s = smooth(sac_rew, SMOOTH) if len(sac_rew) > 0 else []
ppo_succ_s = smooth(ppo_succ, SMOOTH) if len(ppo_succ) > 0 else []
sac_succ_s = smooth(sac_succ, SMOOTH) if len(sac_succ) > 0 else []

plt.figure(figsize=(14, 5))

# Plot 1: Reward
plt.subplot(1, 2, 1)
if len(ppo_rew) > 0:
    plt.plot(ppo_time_r, ppo_rew, alpha=0.3, color='#FF6B6B')
    plt.plot(ppo_time_r, ppo_rew_s, label='PPO', color='#FF6B6B', linewidth=2)
if len(sac_rew) > 0:
    plt.plot(sac_time_r, sac_rew, alpha=0.3, color='#4ECDC4')
    plt.plot(sac_time_r, sac_rew_s, label='SAC', color='#4ECDC4', linewidth=2)

plt.title('Recompensa Episódica Média', fontsize=14, fontweight='bold')
plt.xlabel('Tempo de Relógio (Horas)', fontsize=12)
plt.ylabel('Recompensa', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Plot 2: Success Rate
plt.subplot(1, 2, 2)
if len(ppo_succ) > 0:
    plt.plot(ppo_time_s, ppo_succ * 100, alpha=0.3, color='#FF6B6B')
    plt.plot(ppo_time_s, ppo_succ_s * 100, label='PPO', color='#FF6B6B', linewidth=2)
    idx = np.argmax(ppo_succ_s >= 0.90)
    if ppo_succ_s[idx] >= 0.90:
        plt.plot(ppo_time_s[idx], ppo_succ_s[idx]*100, 'o', color='#FF6B6B', markersize=8)
        plt.axvline(x=ppo_time_s[idx], color='#FF6B6B', linestyle=':', alpha=0.8)
        # Fix: since PPO completes at ~6h, place the label dynamically
        plt.annotate(f"90% ({ppo_time_s[idx]:.1f}h)", xy=(ppo_time_s[idx], ppo_succ_s[idx]*100), 
                     xytext=(-40, -25), textcoords='offset points', color='#FF6B6B', fontweight='bold')

if len(sac_succ) > 0:
    plt.plot(sac_time_s, sac_succ * 100, alpha=0.3, color='#4ECDC4')
    plt.plot(sac_time_s, sac_succ_s * 100, label='SAC', color='#4ECDC4', linewidth=2)
    idx = np.argmax(sac_succ_s >= 0.90)
    if sac_succ_s[idx] >= 0.90:
        plt.plot(sac_time_s[idx], sac_succ_s[idx]*100, 'o', color='#4ECDC4', markersize=8)
        plt.axvline(x=sac_time_s[idx], color='#4ECDC4', linestyle=':', alpha=0.8)
        plt.annotate(f"90% ({sac_time_s[idx]:.1f}h)", xy=(sac_time_s[idx], sac_succ_s[idx]*100), 
                     xytext=(10, -15), textcoords='offset points', color='#4ECDC4', fontweight='bold')

plt.title('Taxa de Sucesso (%)', fontsize=14, fontweight='bold')
plt.xlabel('Tempo de Relógio (Horas)', fontsize=12)
plt.ylabel('Sucesso (%)', fontsize=12)
plt.ylim(0, 105)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()

save_dir = r'c:\Projetos\TCC\TG1_Thiago_200043919__UnB_\unbtex-example\figuras'
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, 'curvas_tempo.pdf'), format='pdf', bbox_inches='tight')
plt.savefig(os.path.join(save_dir, 'curvas_tempo.png'), format='png', dpi=300, bbox_inches='tight')
print("Graficos de tempo salvos com sucesso!")
