import json
import matplotlib.pyplot as plt
import numpy as np
import os

with open("scratch/benchmark_results.json", "r") as f:
    data = json.load(f)

cpu_envs = np.array([int(k) for k in data["cpu"].keys()])
cpu_fps = np.array(list(data["cpu"].values()))

gpu_envs = np.array([int(k) for k in data["gpu"].keys()])
gpu_fps = np.array(list(data["gpu"].values()))

plt.figure(figsize=(9, 6))

# GPU Line
plt.plot(gpu_envs, gpu_fps, marker='o', linewidth=2.5, markersize=8, color='#4ECDC4', label='MuJoCo Warp (GPU Tensor)', zorder=3)

# CPU Line
plt.plot(cpu_envs, cpu_fps, marker='s', linewidth=2.5, markersize=8, color='#FF6B6B', label='Multi-core (CPU)', zorder=3)

# Fill between for effect
plt.fill_between(gpu_envs, gpu_fps, alpha=0.1, color='#4ECDC4')

# Log scale for X axis because of the massive range (1 to 8192)
plt.xscale('log', base=2)

# Styling
plt.title('Curva de Escalabilidade de FPS por Ambientes Simultâneos', fontsize=14, pad=15, fontweight='bold')
plt.xlabel('Número de Ambientes Paralelos (Escala Log 2)', fontsize=12, fontweight='bold')
plt.ylabel('Taxa de Amostragem Físico (FPS)', fontsize=12, fontweight='bold')

plt.grid(True, which="both", ls="--", alpha=0.4, zorder=0)
plt.legend(fontsize=11, loc='upper left')

# Annotate some key points
# GPU Max
max_gpu_idx = np.argmax(gpu_fps)
plt.annotate(f'Pico GPU:\n{int(gpu_fps[max_gpu_idx])} it/s', 
             xy=(gpu_envs[max_gpu_idx], gpu_fps[max_gpu_idx]), 
             xytext=(-60, -30), textcoords='offset points', 
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='#2C3E50'),
             fontsize=10, fontweight='bold', color='#2C3E50')

# CPU Max
max_cpu_idx = np.argmax(cpu_fps)
plt.annotate(f'Saturação CPU:\n{int(cpu_fps[max_cpu_idx])} it/s', 
             xy=(cpu_envs[max_cpu_idx], cpu_fps[max_cpu_idx]), 
             xytext=(20, 20), textcoords='offset points', 
             arrowprops=dict(arrowstyle="->", color='#C0392B'),
             fontsize=10, fontweight='bold', color='#C0392B')

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()

os.makedirs(r'c:\Projetos\TCC\TG1_Thiago_200043919__UnB_\unbtex-example\figuras', exist_ok=True)
plt.savefig(r'c:\Projetos\TCC\TG1_Thiago_200043919__UnB_\unbtex-example\figuras\grafico_fps.pdf', format='pdf', bbox_inches='tight')
plt.savefig(r'c:\Projetos\TCC\TG1_Thiago_200043919__UnB_\unbtex-example\figuras\grafico_fps.png', format='png', dpi=300, bbox_inches='tight')
print("Gráficos salvos com sucesso!")
