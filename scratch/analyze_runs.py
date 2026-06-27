import os
import json
import glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

base_dir = r"c:\Projetos\TCC\active_push_triage\ur3_push_data\rl"

runs = glob.glob(os.path.join(base_dir, "*", "logs"))

results = []

for log_dir in runs:
    run_dir = os.path.dirname(log_dir)
    run_name = os.path.basename(run_dir)
    
    config_path = os.path.join(log_dir, "config.txt")
    cfg = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
        except:
            pass
            
    # Load tensorboard data
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    success_tag = None
    if 'eval/success_rate' in tags:
        success_tag = 'eval/success_rate'
    elif 'rollout/success_rate' in tags:
        success_tag = 'rollout/success_rate'
        
    max_success = 0.0
    steps_to_90 = -1
    if success_tag:
        events = ea.Scalars(success_tag)
        vals = [e.value for e in events]
        steps = [e.step for e in events]
        if len(vals) > 0:
            max_success = max(vals)
            idx_90 = np.argmax(np.array(vals) >= 0.90)
            if vals[idx_90] >= 0.90:
                steps_to_90 = steps[idx_90]
                
    # Extrair parametros importantes
    algo = cfg.get("algorithm", "unknown")
    num_train = cfg.get("numTrain", -1)
    num_stacked = cfg.get("numStackedObs", 0)
    batch_size = cfg.get("batchSize", -1)
    sac_batch = cfg.get("sacBatchSize", -1)
    n_steps = cfg.get("nSteps", -1)
    sac_grad = cfg.get("sacGradientSteps", -1)
    
    results.append({
        "name": run_name,
        "algo": algo,
        "envs": num_train,
        "stack": num_stacked,
        "batch": batch_size if algo=="ppo" else sac_batch,
        "n_steps": n_steps if algo=="ppo" else sac_grad,
        "max_succ": max_success,
        "steps_to_90": steps_to_90
    })

# Print out grouped by algo, sorted by name
print("=== PPO RUNS ===")
for r in sorted([r for r in results if "ppo" in r["algo"].lower()], key=lambda x: x["name"]):
    print(f"{r['name']}: envs={r['envs']}, stack={r['stack']}, batch={r['batch']}, n_steps={r['n_steps']} | max_succ={r['max_succ']:.2f}, steps_to_90={r['steps_to_90']}")

print("\n=== SAC RUNS ===")
for r in sorted([r for r in results if "sac" in r["algo"].lower()], key=lambda x: x["name"]):
    print(f"{r['name']}: envs={r['envs']}, stack={r['stack']}, batch={r['batch']}, sac_grads={r['n_steps']} | max_succ={r['max_succ']:.2f}, steps_to_90={r['steps_to_90']}")

print("\n=== OTHER RUNS ===")
for r in sorted([r for r in results if r["algo"] not in ["ppo", "sac", "SAC"] and ("ppo" not in r["algo"].lower() and "sac" not in r["algo"].lower())], key=lambda x: x["name"]):
    print(f"{r['name']}: algo={r['algo']}, max_succ={r['max_succ']:.2f}, steps_to_90={r['steps_to_90']}")
