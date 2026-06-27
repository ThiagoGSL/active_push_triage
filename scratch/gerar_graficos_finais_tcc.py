import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import os
from matplotlib.ticker import FuncFormatter

# CAMINHOS FINAIS DOS TREINAMENTOS
ppo_path = r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl\MujocoUR3PushEnv_ppo_teste_fps_mesa_20260625_223152\logs"
sac_path = r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl\MujocoUR3PushEnv_sac_teste_fps_mesa_20260626_015332\logs"

def get_data(path, tag):
    if not os.path.exists(path): return np.array([]), np.array([])
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

    steps = [e.step for e in events]
    vals = [e.value for e in events]
    return np.array(steps), np.array(vals)

def get_time_data(path, tag):
    if not os.path.exists(path): return np.array([]), np.array([])
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
    
    try:
        start_time = min([ea.Scalars(t)[0].wall_time for t in ea.Tags()['scalars'] if ea.Scalars(t)])
    except:
        start_time = events[0].wall_time
        
    times_h = [(e.wall_time - start_time)/3600.0 for e in events]
    vals = [e.value for e in events]
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

def millions_formatter(x, pos):
    return f'{x / 1e6:.1f}M'

print("Lendo os dados do TensorBoard...")
ppo_steps_succ, ppo_succ = get_data(ppo_path, 'eval/success_rate')
sac_steps_succ, sac_succ = get_data(sac_path, 'eval/success_rate')

ppo_time_succ, ppo_succ_t = get_time_data(ppo_path, 'eval/success_rate')
sac_time_succ, sac_succ_t = get_time_data(sac_path, 'eval/success_rate')

ppo_steps_rew, ppo_rew = get_data(ppo_path, 'eval/mean_reward')
sac_steps_rew, sac_rew = get_data(sac_path, 'eval/mean_reward')

ppo_time_fps, ppo_fps = get_time_data(ppo_path, 'time/fps')
sac_time_fps, sac_fps = get_time_data(sac_path, 'time/fps')

# CORTAR DADOS DO SAC EM 3 HORAS
if len(sac_time_succ) > 0:
    mask = sac_time_succ <= 3.0
    sac_steps_succ = sac_steps_succ[mask]
    sac_succ = sac_succ[mask]
    sac_time_succ = sac_time_succ[mask]
    sac_succ_t = sac_succ_t[mask]
    if len(sac_steps_rew) > 0:
        sac_steps_rew = sac_steps_rew[:len(sac_steps_succ)]
        sac_rew = sac_rew[:len(sac_steps_succ)]
        
if len(sac_time_fps) > 0:
    mask_fps = sac_time_fps <= 3.0
    sac_time_fps = sac_time_fps[mask_fps]
    sac_fps = sac_fps[mask_fps]

# AMOSTRAGEM MENOR PARA O SAC (subsample para suavizar curvas densas)
if len(sac_succ) > 100:
    step_size = max(1, len(sac_succ) // 100)
    sac_steps_succ = sac_steps_succ[::step_size]
    sac_succ = sac_succ[::step_size]
    sac_time_succ = sac_time_succ[::step_size]
    sac_succ_t = sac_succ_t[::step_size]
    
if len(sac_rew) > 100:
    step_size = max(1, len(sac_rew) // 100)
    sac_steps_rew = sac_steps_rew[::step_size]
    sac_rew = sac_rew[::step_size]

SMOOTH = 0.8
ppo_succ_s = smooth(ppo_succ, SMOOTH) if len(ppo_succ) > 0 else []
sac_succ_s = smooth(sac_succ, SMOOTH) if len(sac_succ) > 0 else []

ppo_succ_ts = smooth(ppo_succ_t, SMOOTH) if len(ppo_succ_t) > 0 else []
sac_succ_ts = smooth(sac_succ_t, SMOOTH) if len(sac_succ_t) > 0 else []

ppo_rew_s = smooth(ppo_rew, SMOOTH) if len(ppo_rew) > 0 else []
sac_rew_s = smooth(sac_rew, SMOOTH) if len(sac_rew) > 0 else []

plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(18, 12))

def plot_annotations(ax, x_arr, x_steps_arr, y_arr_raw, rew_raw, steps_rew, color, label_prefix, is_time=False, y_offset=-40):
    if len(y_arr_raw) > 0:
        idx_90 = np.argmax(y_arr_raw >= 0.90)
        if y_arr_raw[idx_90] >= 0.90:
            val_x = x_arr[idx_90]
            val_y = y_arr_raw[idx_90] * 100
            ax.plot(val_x, val_y, '*', color='gold', markersize=15, markeredgecolor='black', zorder=5)
            x_label = f"{val_x:.1f}h" if is_time else f"{val_x/1e6:.1f}M"
            ax.annotate(f"90% {label_prefix}\n({x_label})", 
                         xy=(val_x, val_y), 
                         xytext=(-40, y_offset), textcoords='offset points', color='black', fontweight='bold',
                         arrowprops=dict(arrowstyle="->", color='black'), zorder=5)

    if len(rew_raw) > 0 and len(steps_rew) > 0:
        # Pega a melhor recompensa baseada no index de rew_raw
        idx_best_rew = np.argmax(rew_raw)
        best_step_abs = steps_rew[idx_best_rew]  # Isso é o passo exato da melhor recompensa
        
        # Encontra o índice em x_steps_arr mais próximo de best_step_abs
        idx_best_mapped = np.argmin(np.abs(np.array(x_steps_arr) - best_step_abs))
        
        if idx_best_mapped < len(x_arr) and idx_best_mapped < len(y_arr_raw):
            val_x = x_arr[idx_best_mapped]
            val_y = y_arr_raw[idx_best_mapped] * 100
            ax.plot(val_x, val_y, 'o', color=color, markersize=10, markeredgecolor='white', zorder=5)
            x_label = f"{val_x:.1f}h" if is_time else f"{val_x/1e6:.1f}M"
            ax.annotate(f"Melhor {label_prefix}\n({x_label})", 
                         xy=(val_x, val_y), 
                         xytext=(10, y_offset), textcoords='offset points', color=color, fontweight='bold',
                         arrowprops=dict(arrowstyle="->", color=color), zorder=5)

# ==========================================
# PLOT 1: Sucesso vs STEPS (com Marcações)
# ==========================================
ax1 = plt.subplot(2, 2, 1)
if len(ppo_succ) > 0:
    ax1.plot(ppo_steps_succ, ppo_succ * 100, alpha=0.2, color='#FF6B6B')
    ax1.plot(ppo_steps_succ, ppo_succ_s * 100, label='PPO (Warp GPU)', color='#FF6B6B', linewidth=2.5)
    plot_annotations(ax1, ppo_steps_succ, ppo_steps_succ, ppo_succ, ppo_rew, ppo_steps_rew, '#A00000', 'PPO', is_time=False, y_offset=-40)

if len(sac_succ) > 0:
    ax1.plot(sac_steps_succ, sac_succ * 100, alpha=0.2, color='#4ECDC4')
    ax1.plot(sac_steps_succ, sac_succ_s * 100, label='SAC (Warp GPU)', color='#4ECDC4', linewidth=2.5)
    plot_annotations(ax1, sac_steps_succ, sac_steps_succ, sac_succ, sac_rew, sac_steps_rew, '#006D66', 'SAC', is_time=False, y_offset=20)

ax1.set_title('Evolução do Aprendizado (Por Passos)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Passos de Simulação', fontsize=12)
ax1.set_ylabel('Taxa de Sucesso (%)', fontsize=12)
ax1.xaxis.set_major_formatter(FuncFormatter(millions_formatter))
ax1.legend(loc='lower right')

# ==========================================
# PLOT 2: Sucesso vs WALL TIME
# ==========================================
ax2 = plt.subplot(2, 2, 2)
if len(ppo_succ_t) > 0:
    ax2.plot(ppo_time_succ, ppo_succ_t * 100, alpha=0.2, color='#FF6B6B')
    ax2.plot(ppo_time_succ, ppo_succ_ts * 100, label='PPO (Warp GPU)', color='#FF6B6B', linewidth=2.5)
    plot_annotations(ax2, ppo_time_succ, ppo_steps_succ, ppo_succ_t, ppo_rew, ppo_steps_rew, '#A00000', 'PPO', is_time=True, y_offset=-40)

if len(sac_succ_t) > 0:
    ax2.plot(sac_time_succ, sac_succ_t * 100, alpha=0.2, color='#4ECDC4')
    ax2.plot(sac_time_succ, sac_succ_ts * 100, label='SAC (Warp GPU)', color='#4ECDC4', linewidth=2.5)
    plot_annotations(ax2, sac_time_succ, sac_steps_succ, sac_succ_t, sac_rew, sac_steps_rew, '#006D66', 'SAC', is_time=True, y_offset=20)

ax2.set_title('Eficácia Temporal (Tempo de Relógio)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Tempo Real de Treinamento (Horas)', fontsize=12)
ax2.set_ylabel('Taxa de Sucesso (%)', fontsize=12)
ax2.legend(loc='lower right')

# ==========================================
# PLOT 3: Iterações por Segundo (FPS) vs WALL TIME
# ==========================================
ax3 = plt.subplot(2, 1, 2)
if len(ppo_fps) > 0:
    ax3.plot(ppo_time_fps, smooth(ppo_fps, 0.5), label='PPO (Warp GPU)', color='#FF6B6B', linewidth=2.5)
if len(sac_fps) > 0:
    ax3.plot(sac_time_fps, smooth(sac_fps, 0.5), label='SAC (Warp GPU)', color='#4ECDC4', linewidth=2.5)

ax3.set_title('Velocidade Computacional Bruta (Através do Tempo)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Tempo Real de Treinamento (Horas)', fontsize=12)
ax3.set_ylabel('FPS (Passos por Segundo)', fontsize=12)
ax3.fill_between(ppo_time_fps, 0, smooth(ppo_fps, 0.5), alpha=0.1, color='#FF6B6B')
if len(sac_fps) > 0:
    ax3.fill_between(sac_time_fps, 0, smooth(sac_fps, 0.5), alpha=0.1, color='#4ECDC4')
ax3.legend(loc='lower right')

plt.tight_layout()
out_dir = r"c:\Projetos\TCC\TG1_Thiago_200043919__UnB_\unbtex-example\figuras"
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "graficos_finais_tcc.pdf"), format="pdf", bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "graficos_finais_tcc.png"), format="png", dpi=300, bbox_inches='tight')
print("Gráficos salvos com sucesso em figuras/graficos_finais_tcc!")
