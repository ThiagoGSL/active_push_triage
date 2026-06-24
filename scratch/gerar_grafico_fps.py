import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['Multi-core CPU\n(16 instâncias)', 'MuJoCo Warp\n(1024 instâncias)']
fps = [493, 29606]
colors = ['#FF6B6B', '#4ECDC4']

plt.figure(figsize=(7, 5))
bars = plt.bar(labels, fps, color=colors, width=0.5)

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 500, f"{int(yval)} it/s", ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylabel('Iterações por Segundo (FPS)', fontsize=12, fontweight='bold')
plt.title('Taxa de Amostragem do Motor Físico', fontsize=14, pad=15, fontweight='bold')
plt.ylim(0, 35000)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Make it look clean
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Save
import os
os.makedirs(r'c:\Projetos\TCC\TG1_Thiago_200043919__UnB_\unbtex-example\figuras', exist_ok=True)
plt.tight_layout()
plt.savefig(r'c:\Projetos\TCC\TG1_Thiago_200043919__UnB_\unbtex-example\figuras\grafico_fps.pdf', format='pdf', bbox_inches='tight')
plt.savefig(r'c:\Projetos\TCC\TG1_Thiago_200043919__UnB_\unbtex-example\figuras\grafico_fps.png', format='png', dpi=300, bbox_inches='tight')
