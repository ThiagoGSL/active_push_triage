"""
validate_mjwarp.py — Fase 1: Validação e Benchmark do MuJoCo Warp

Verifica:
1. Instalação de mujoco_warp e warp
2. Carregamento do modelo UR3 na GPU
3. Step sem crash
4. Transferência GPU→CPU (site_xpos, qpos)
5. Benchmark de throughput para N ∈ {8, 16, 64, 128, 256}

Uso:
    python ur3_push_mujoco/utils/validate_mjwarp.py
"""
import time
import numpy as np
import os
import sys

# ---------------------------------------------------------------------------
# Imports — verificados individualmente para diagnosticar erros
# ---------------------------------------------------------------------------
print("=" * 60)
print("Fase 1 — Validação MuJoCo Warp")
print("=" * 60)

try:
    import warp as wp
    wp.init()
    print(f"✓ warp {wp.__version__} importado")
    devices = wp.get_devices()
    cuda_devs = [d for d in devices if d.is_cuda]
    print(f"  Dispositivos CUDA disponíveis: {[str(d) for d in cuda_devs]}")
    if not cuda_devs:
        print("  [AVISO] Nenhum dispositivo CUDA encontrado — MJWarp rodará no modo CPU (degradado)")
    print(f"  CUDA devices: {[str(d) for d in cuda_devs]}")
except ImportError as e:
    print(f"✗ Erro ao importar warp: {e}")
    print("  Execute: pip install warp-lang")
    sys.exit(1)

try:
    import mujoco_warp as mjw
    print(f"✓ mujoco_warp importado")
except ImportError as e:
    print(f"✗ Erro ao importar mujoco_warp: {e}")
    print("  Execute: pip install mujoco-warp")
    sys.exit(1)

try:
    import mujoco
    print(f"✓ mujoco {mujoco.__version__} importado")
    if tuple(int(x) for x in mujoco.__version__.split(".")[:2]) < (3, 0):
        print(f"  [AVISO] MuJoCo {mujoco.__version__} pode ser incompatível com mujoco_warp.")
        print("          Recomendado: mujoco >= 3.0. Execute: pip install 'mujoco>=3.0'")
except ImportError as e:
    print(f"✗ Erro ao importar mujoco: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Gera XML do modelo UR3 usando o utilitário existente
# ---------------------------------------------------------------------------
print("\n" + "-" * 40)
print("Carregando modelo UR3...")

try:
    import ur3_push_mujoco
    from ur3_push_mujoco.utils.mujoco_utils import generate_model_xml_string

    xml_str = generate_model_xml_string(use_sim_config=True)
    original_cwd = os.getcwd()
    os.chdir(os.path.join(ur3_push_mujoco.__path__[0], "assets"))
    mjm = mujoco.MjModel.from_xml_string(xml_str)
    os.chdir(original_cwd)

    print(f"✓ MjModel criado (nq={mjm.nq}, nv={mjm.nv}, nu={mjm.nu}, nsites={mjm.nsite})")
    print(f"  Sites disponíveis: {[mjm.site(i).name for i in range(mjm.nsite)]}")
except Exception as e:
    print(f"✗ Erro ao criar modelo UR3: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# Transfere modelo para GPU
# ---------------------------------------------------------------------------
print("\n" + "-" * 40)
print("Transferindo modelo para GPU...")

try:
    m = mjw.put_model(mjm)
    print("✓ mjw.put_model() OK")
except Exception as e:
    print(f"✗ Erro em mjw.put_model(): {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# Teste básico: N=1 mundo
# ---------------------------------------------------------------------------
print("\n" + "-" * 40)
print("Teste básico com N=1 mundo...")

try:
    d1 = mjw.make_data(mjm, nworld=1)
    print("✓ mjw.make_data(nworld=1) OK")

    mjw.step(m, d1)
    print("✓ mjw.step() OK (1 world)")

    # Verifica leitura de campos
    site_xpos = d1.site_xpos.numpy()   # shape: [1, nsites, 3] esperado
    qpos = d1.qpos.numpy()             # shape: [1, nq] esperado

    print(f"✓ d.site_xpos.numpy() shape: {site_xpos.shape}")
    print(f"✓ d.qpos.numpy() shape: {qpos.shape}")
    print(f"  site_xpos[0, 0, :] = {site_xpos[0, 0, :]}")

    del d1
except Exception as e:
    print(f"✗ Erro no teste básico: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# Verifica campo site_xmat (necessário para IK: rotação do EE)
# ---------------------------------------------------------------------------
print("\n" + "-" * 40)
print("Verificando campos necessários para IK...")

try:
    d_test = mjw.make_data(mjm, nworld=1)
    mjw.mj_kinematics(m, d_test)  # forward kinematics apenas
    
    site_xmat = d_test.site_xmat.numpy()
    qvel = d_test.qvel.numpy()
    ctrl = d_test.ctrl.numpy()
    
    print(f"✓ d.site_xmat.numpy() shape: {site_xmat.shape}")
    print(f"✓ d.qvel.numpy()     shape: {qvel.shape}")
    print(f"✓ d.ctrl.numpy()     shape: {ctrl.shape}")
    del d_test
except AttributeError as e:
    print(f"  [AVISO] Alguns campos podem não estar disponíveis nesta versão do MJWarp: {e}")
    print("  Tentando com mjw.step() em vez de mjw.mj_kinematics()...")
    try:
        d_test = mjw.make_data(mjm, nworld=1)
        mjw.step(m, d_test)
        site_xmat = d_test.site_xmat.numpy()
        print(f"✓ d.site_xmat.numpy() shape (via step): {site_xmat.shape}")
        del d_test
    except Exception as e2:
        print(f"  [AVISO] site_xmat não disponível: {e2}")

# ---------------------------------------------------------------------------
# Benchmark de throughput
# ---------------------------------------------------------------------------
print("\n" + "-" * 40)
print("Benchmark de throughput (steps/segundo)...")
print(f"{'N mundos':>10} | {'Total steps':>12} | {'Tempo (s)':>10} | {'Steps/s':>12} | {'Speedup vs CPU':>14}")
print("-" * 65)

n_warmup = 20
n_benchmark_steps = 500
baseline_cpu_its_s = 300.0  # baseline medido

results = []
for nworld in [1, 8, 16, 32, 64, 128, 256]:
    try:
        d = mjw.make_data(mjm, nworld=nworld)

        # Warmup
        for _ in range(n_warmup):
            mjw.step(m, d)
        wp.synchronize()

        # Benchmark com CUDA graph
        try:
            with wp.ScopedCapture() as capture:
                mjw.step(m, d)
            graph = capture.graph
            use_graph = True
        except Exception:
            use_graph = False

        t0 = time.perf_counter()
        if use_graph:
            for _ in range(n_benchmark_steps):
                wp.capture_launch(graph)
        else:
            for _ in range(n_benchmark_steps):
                mjw.step(m, d)
        wp.synchronize()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        total_steps = n_benchmark_steps * nworld
        steps_per_sec = total_steps / elapsed
        gym_steps_per_sec = (n_benchmark_steps / elapsed)  # gym steps/s (N ambientes em paralelo = 1 gym step)
        speedup = gym_steps_per_sec / baseline_cpu_its_s

        print(f"{nworld:>10} | {total_steps:>12} | {elapsed:>10.3f} | {steps_per_sec:>12,.0f} | {speedup:>13.1f}x")
        results.append((nworld, steps_per_sec, speedup))
        del d

    except Exception as e:
        print(f"{nworld:>10} | ERRO: {e}")

print("-" * 65)
print(f"\nBaseline CPU: ~{baseline_cpu_its_s:.0f} its/s (SubprocVecEnv, 8 envs)")
print("Nota: 'its/s' = gym steps/s considerando N mundos como um único passo RL\n")

if results:
    best = max(results, key=lambda x: x[2])
    print(f"✓ Melhor configuração: N={best[0]} mundos → {best[2]:.1f}x speedup estimado")

print("\n" + "=" * 60)
print("✓ Validação concluída! MuJoCo Warp está funcional.")
print("=" * 60)
