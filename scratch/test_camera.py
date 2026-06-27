import os
import mujoco
from PIL import Image
import numpy as np

# Vamos importar o ambiente real para ter a mesma inicialização do treinamento/avaliação
from ur3_push_rl_sb3.make_envs import make_warp_env

# Cria o ambiente warp idêntico ao da avaliação
_, eval_env = make_warp_env(
    num_train=1,
    max_episode_steps=100,
    sparse_reward=False,
    ee_to_obj_reward_scale=0.0,
    action_scaling_factor=0.05,
    n_substeps=120,
    seed=42,
    torque_penalty_scale=0.001,
    action_rate_penalty_scale=0.05,
    success_bonus=0.0,
    early_termination_on_success=0,
    randomize_initial_joints=False,
    manipulability_reward_scale=0.01,
    manipulability_metric="yoshikawa"
)

warp_wenv = eval_env

# Faz o reset para o ambiente posicionar tudo perfeitamente (Mesa, robô, IK, etc)
obs = eval_env.reset()

m = warp_wenv._wenv._mjm
d = warp_wenv._wenv._cpu_data

# Força o target_xypos a atualizar no MJData
target_xy = warp_wenv._wenv._target_xypos[0]
body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target')
if body_id >= 0:
    m.body_pos[body_id][0] = target_xy[0]
    m.body_pos[body_id][1] = target_xy[1]

# Sincroniza estado real do Warp para o CPU
d.qpos[:] = warp_wenv._wenv._d_gpu.qpos.numpy()[0]
d.qvel[:] = warp_wenv._wenv._d_gpu.qvel.numpy()[0]
if hasattr(warp_wenv._wenv._d_gpu, 'mocap_pos'):
    d.mocap_pos[:] = warp_wenv._wenv._d_gpu.mocap_pos.numpy()[0]
    d.mocap_quat[:] = warp_wenv._wenv._d_gpu.mocap_quat.numpy()[0]

mujoco.mj_forward(m, d)

# Renderiza exatamente a cena do eval (A posição/target agora está hardcoded no XML)
renderer = mujoco.Renderer(m, 480, 640)
renderer.update_scene(d, camera="eval_camera")
pixels = renderer.render()

Image.fromarray(pixels).save("camera_test.png")
print("Imagem salva em camera_test.png com o estado exato da engine Warp!")
