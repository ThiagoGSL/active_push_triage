"""
ur3_push_simple_warp_env.py — Fase 4: Ambiente UR3 Push para MuJoCo Warp

Implementa a lógica de N mundos em paralelo para treinamento com MJWarp:
  - Reset de estado (qpos do robô + pose do objeto/target) para N mundos
  - Observações em batch [N, 8]
  - Recompensas em batch [N]
  - Terminações em batch [N]

Design:
  - O modelo MuJoCo é criado UMA VEZ e passado para mjw.put_model()
  - Poses de objeto/target são escritas diretamente em d.qpos na GPU via warp arrays
  - Sem reconstrução de XML — incompatível com MJWarp
  - Randomização de parâmetros físicos (massa, fricção) é desabilitada por padrão
    (eles são propriedades do MjModel estático, compartilhado entre todos os mundos)
"""

import os
import numpy as np
import mujoco
import warp as wp
import mujoco_warp as mjw
import ur3_push_mujoco
from gymnasium_robotics.utils import rotations

from ur3_push_mujoco.utils import mujoco_utils, rotations_utils
from ur3_push_mujoco.utils.mujoco_utils import MuJoCoUR3PushControllerBatched


# ---------------------------------------------------------------------------
# Configuração padrão do robô (joint names → qpos iniciais)
# ---------------------------------------------------------------------------
_INITIAL_QPOS = {
    "shoulder_pan_joint":        0.0,
    "shoulder_lift_joint":      -1.5708,
    "elbow_joint":               1.5708,
    "wrist_1_joint":            -1.5708,
    "wrist_2_joint":            -1.5708,
    "wrist_3_joint":             0.0,
    "finger_joint":              0.700,
    "right_outer_knuckle_joint": -0.700,
    "left_inner_knuckle_joint":  -0.700,
    "right_inner_knuckle_joint": -0.700,
    "left_inner_finger_joint":   0.700,
    "right_inner_finger_joint":  0.700,
}

# Limites do workspace (mesmo que MuJoCoUR3PushController)
_MIN_EE_XY = np.array([0.1, -0.25])
_MAX_EE_XY = np.array([0.65, 0.25])
_INITIAL_EE_ZPOS = 0.045
_HEIGHT_TABLE = 0.02

# Spawn range relativo ao EE inicial (mesmo que MuJoCoUR3PushSimpleEnv)
_RANGE_X_POS = np.array([0.07, 0.27])
_RANGE_Y_POS = np.array([-0.12, 0.12])
_THRESHOLD_POS = 0.05


class MuJoCoUR3PushSimpleWarpEnv:
    """Ambiente UR3 Push para MuJoCo Warp — N mundos em paralelo.

    Não é um `gymnasium.Env`. É usado internamente pelo `WarpVecEnv`
    para gerenciar o estado de N mundos e calcular obs/reward/done.

    Args:
        nworld: Número de mundos em paralelo na GPU.
        n_substeps: Número de substeps de física por gym step.
        sparse_reward: Se True, usa recompensa esparsa (-1/0).
        ee_to_obj_reward_scale: Peso do reward de aproximação EE→objeto.
        action_scaling_factor: Fator de escala das ações.
        seed: Seed do RNG para reprodutibilidade.
    """

    def __init__(self,
                 nworld: int = 64,
                 n_substeps: int = 40,
                 sparse_reward: bool = True,
                 ee_to_obj_reward_scale: float = 0.2,
                 action_scaling_factor: float = 0.6,
                 seed: int = 0):

        self.nworld = nworld
        self.n_substeps = n_substeps
        self.sparse_reward = sparse_reward
        self.ee_to_obj_reward_scale = ee_to_obj_reward_scale
        self.action_scaling_factor = action_scaling_factor
        self.rng = np.random.default_rng(seed)

        # --- Carrega modelo CPU (UMA VEZ) ---
        xml = mujoco_utils.generate_model_xml_string(
            robot_gravity_compensation=1, use_sim_config=True
        )
        original_cwd = os.getcwd()
        os.chdir(os.path.join(ur3_push_mujoco.__path__[0], "assets"))
        self._mjm = mujoco.MjModel.from_xml_string(xml)
        os.chdir(original_cwd)

        self._cpu_data = mujoco.MjData(self._mjm)

        # IDs úteis
        self._obj_joint_id   = self._mjm.joint("object_joint").id
        self._obj_joint_qadr = self._mjm.joint("object_joint").qposadr[0]
        self._obj_site_id    = self._mjm.site("object_site").id
        self._ee_site_id     = self._mjm.site("pinch_site").id
        # O target e um body FIXO no modelo MuJoCo (sem freejoint).
        # Sua posicao nao pode ser alterada via qpos na GPU.
        # A posicao por mundo e armazenada em _target_xypos (numpy) e usada
        # somente para calcular obs e reward — sem escrita na GPU.
        self._target_body_id = self._mjm.body("target").id

        # --- Posição inicial do robô (qpos base) ---
        self._initial_robot_qpos = self._build_initial_qpos()

        # --- Posição inicial do EE (para calcular spawn range) ---
        self._cpu_data.qpos[:] = self._initial_robot_qpos
        mujoco.mj_forward(self._mjm, self._cpu_data)
        self.initial_ee_xypos = self._cpu_data.site_xpos[self._ee_site_id][:2].copy()

        # --- GPU: carrega modelo e cria batch de dados ---
        self._m_gpu = mjw.put_model(self._mjm)
        self._d_gpu = mjw.make_data(self._mjm, nworld=nworld)

        # --- Controlador IK batched ---
        self.controller = MuJoCoUR3PushControllerBatched(
            model=self._mjm,
            robot_name="ur3",
            ee_site_name="pinch_site",
            initial_ee_zpos=_INITIAL_EE_ZPOS,
            min_ee_xy_pos=_MIN_EE_XY,
            max_ee_xy_pos=_MAX_EE_XY,
            use_sim_config=True,
        )

        # --- CUDA graph para substep loop ---
        # Será (re)capturado após o primeiro reset, quando o estado estiver definido
        self._cuda_graph = None
        self._graph_captured = False

        # --- Estado de episódio ---
        # Pose do target por mundo [N, 2] (xy) para calcular obs/reward
        self._target_xypos = np.zeros((nworld, 2))
        self._elapsed_steps = np.zeros(nworld, dtype=np.int32)

    # -----------------------------------------------------------------------
    # Inicialização
    # -----------------------------------------------------------------------

    def _build_initial_qpos(self) -> np.ndarray:
        """Constrói o vetor qpos inicial para todos os joints (robô + obj + target)."""
        qpos = np.zeros(self._mjm.nq)
        for name, val in _INITIAL_QPOS.items():
            try:
                jid = self._mjm.joint(name).id
                qadr = self._mjm.joint(name).qposadr[0]
                qpos[qadr] = val
            except (ValueError, KeyError):
                pass  # joint não existe nesta versão do modelo
        return qpos

    def _capture_cuda_graph(self):
        """Captura o CUDA graph para o substep loop (rápido após warmup)."""
        with wp.ScopedCapture() as capture:
            for _ in range(self.n_substeps):
                mjw.step(self._m_gpu, self._d_gpu)
        self._cuda_graph = capture.graph
        self._graph_captured = True

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def reset(self, world_ids=None) -> dict:
        """Reseta mundos especificados (ou todos se world_ids=None).

        Randomiza:
        - Pose do objeto (xy, z-rotation)
        - Pose do target (xy, z-rotation) — ao menos 10cm do objeto

        Returns:
            obs_batch: dict com 'observation' [N,8], 'achieved_goal' [N,2], 'desired_goal' [N,2]
        """
        if world_ids is None:
            world_ids = np.arange(self.nworld)

        # Lê qpos atual da GPU para modificar
        qpos_all = self._d_gpu.qpos.numpy().copy()  # [N, nq]

        for i in world_ids:
            # Reseta joints do robô
            qpos_all[i] = self._initial_robot_qpos.copy()

            # Amostra pose do objeto
            obj_xy = self._sample_xy_away_from(self.initial_ee_xypos, min_dist=0.1)
            obj_z  = _HEIGHT_TABLE + 0.04 + 0.001  # altura padrao (caixa 4cm)
            obj_zangle = self.rng.uniform(-np.pi, np.pi)
            obj_quat = self._zangle_to_quat(obj_zangle)

            # qpos do objeto (freejoint: [x, y, z, qw, qx, qy, qz])
            qadr = self._obj_joint_qadr
            qpos_all[i, qadr:qadr+3] = [obj_xy[0], obj_xy[1], obj_z]
            qpos_all[i, qadr+3:qadr+7] = obj_quat

            # Amostra posicao do target (>= 10cm do objeto)
            # O target e um body fixo no modelo — so armazenamos a posicao em numpy.
            # A observacao e o reward usam _target_xypos; o visual da GPU fica fixo.
            tgt_xy = self._sample_xy_away_from(obj_xy, min_dist=0.10)
            self._target_xypos[i] = tgt_xy
            self._elapsed_steps[i] = 0

        # Zera velocidades e aplica qpos na GPU
        # MJWarp usa float32 para qpos e qvel
        qpos_f32 = qpos_all.astype(np.float32)
        qvel_zeros = np.zeros(self._d_gpu.qvel.numpy().shape, dtype=np.float32)
        wp.copy(self._d_gpu.qpos, wp.from_numpy(qpos_f32, dtype=wp.float32))
        wp.copy(self._d_gpu.qvel, wp.from_numpy(qvel_zeros, dtype=wp.float32))
        wp.synchronize()

        # Forward para atualizar cinemática na GPU
        mjw.step(self._m_gpu, self._d_gpu)
        wp.synchronize()

        # (Re)captura CUDA graph após mudança de estado
        self._capture_cuda_graph()

        return self._get_obs_batch()

    def _sample_xy_away_from(self, ref_xy: np.ndarray, min_dist: float) -> np.ndarray:
        """Amostra posição xy suficientemente distante de ref_xy."""
        for _ in range(100):
            x = self.initial_ee_xypos[0] + self.rng.uniform(*_RANGE_X_POS)
            y = self.initial_ee_xypos[1] + self.rng.uniform(*_RANGE_Y_POS)
            if np.linalg.norm(np.array([x, y]) - ref_xy) >= min_dist:
                return np.array([x, y])
        # Fallback: ponto fixo
        return ref_xy + np.array([min_dist * 1.5, 0.0])

    @staticmethod
    def _zangle_to_quat(zangle: float) -> np.ndarray:
        """Converte ângulo z (rad) para quaternion MuJoCo [w, x, y, z]."""
        half = zangle / 2.0
        return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])

    # -----------------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------------

    def step(self, actions: np.ndarray):
        """Executa um gym step para N mundos.

        Args:
            actions: [N, 2] — ações do agente (delta x, delta y normalizados)

        Returns:
            obs_batch, rewards, terminateds, truncateds, infos
        """
        # 1. GPU→CPU: lê estado para IK
        site_xpos = self._d_gpu.site_xpos.numpy()  # [N, nsites, 3]
        site_xmat = self._d_gpu.site_xmat.numpy()  # [N, nsites, 3, 3]
        qpos      = self._d_gpu.qpos.numpy()        # [N, nq]

        # 2. CPU: IK vetorizado → ctrl_batch [N, 6]
        scaled_actions = actions * self.action_scaling_factor
        ee_pos_batch = site_xpos[:, self._ee_site_id, :]  # [N, 3]
        ee_pos_d = self.controller.update_desired_pose_batched(
            ee_pos_batch, scaled_actions
        )
        ctrl_batch = self.controller.compute_ctrl_batched(
            qpos, site_xpos, site_xmat, ee_pos_d
        )

        # 3. CPU→GPU: aplica ctrl (apenas joints do braço)
        # MJWarp usa float32 para ctrl
        ctrl_gpu = self._d_gpu.ctrl.numpy().copy()  # [N, nu] float32
        ctrl_gpu[:, self.controller.panda_actuator_ids] = ctrl_batch.astype(np.float32)
        wp.copy(self._d_gpu.ctrl, wp.from_numpy(ctrl_gpu.astype(np.float32), dtype=wp.float32))

        # 4. GPU: substep loop via CUDA graph
        if self._graph_captured:
            wp.capture_launch(self._cuda_graph)
        else:
            for _ in range(self.n_substeps):
                mjw.step(self._m_gpu, self._d_gpu)
        wp.synchronize()

        # 5. Incrementa contagem de steps
        self._elapsed_steps += 1

        # 6. Obs, reward, done
        obs = self._get_obs_batch()
        rewards = self._compute_rewards(obs["achieved_goal"], obs["desired_goal"], site_xpos)
        terminateds = self._compute_terminated(obs["achieved_goal"], obs["desired_goal"])
        truncateds = np.zeros(self.nworld, dtype=bool)
        infos = [{"is_success": bool(terminateds[i])} for i in range(self.nworld)]

        return obs, rewards, terminateds, truncateds, infos

    # -----------------------------------------------------------------------
    # Observações
    # -----------------------------------------------------------------------

    def _get_obs_batch(self) -> dict:
        """Constrói observações para N mundos a partir do estado da GPU.

        Returns:
            dict com:
              'observation':    [N, 8] — [ee_xy(2), obj_xy(2), ee→obj(2), obj→tgt(2)]
              'achieved_goal':  [N, 2] — posição xy do objeto
              'desired_goal':   [N, 2] — posição xy do target
        """
        site_xpos = self._d_gpu.site_xpos.numpy()  # [N, nsites, 3]

        ee_pos  = site_xpos[:, self._ee_site_id,  :2]  # [N, 2]
        obj_pos = site_xpos[:, self._obj_site_id, :2]   # [N, 2]

        # Adiciona ruído Gaussiano leve (replica comportamento do env original)
        obj_pos = obj_pos + self.rng.normal(0, 0.0001, obj_pos.shape)
        ee_pos  = ee_pos  + self.rng.normal(0, 0.0001, ee_pos.shape)

        desired_goal = self._target_xypos.copy()  # [N, 2]

        ee_to_obj    = obj_pos - ee_pos           # [N, 2]
        obj_to_target = desired_goal - obj_pos    # [N, 2]

        observation = np.concatenate([ee_pos, obj_pos, ee_to_obj, obj_to_target], axis=1)  # [N, 8]

        return {
            "observation":   observation,
            "achieved_goal": obj_pos.copy(),
            "desired_goal":  desired_goal,
        }

    # -----------------------------------------------------------------------
    # Reward e Terminação
    # -----------------------------------------------------------------------

    def _compute_rewards(self,
                          achieved_goal: np.ndarray,
                          desired_goal: np.ndarray,
                          site_xpos_pre: np.ndarray) -> np.ndarray:
        """Calcula recompensas para N mundos.

        Args:
            achieved_goal: [N, 2] — posição xy do objeto
            desired_goal:  [N, 2] — posição xy do target
            site_xpos_pre: [N, nsites, 3] — posição dos sites (pré-step, para EE)

        Returns:
            rewards: [N] — float
        """
        dist_obj_tgt = np.linalg.norm(desired_goal - achieved_goal, axis=1)  # [N]

        if self.sparse_reward:
            rewards = -(dist_obj_tgt >= _THRESHOLD_POS).astype(np.float32)
        else:
            rewards = -dist_obj_tgt.astype(np.float32)

            if self.ee_to_obj_reward_scale > 0:
                ee_xy = site_xpos_pre[:, self._ee_site_id, :2]  # [N, 2]
                dist_ee_obj = np.linalg.norm(achieved_goal - ee_xy, axis=1)  # [N]
                rewards -= self.ee_to_obj_reward_scale * dist_ee_obj

        return rewards

    def _compute_terminated(self,
                              achieved_goal: np.ndarray,
                              desired_goal: np.ndarray) -> np.ndarray:
        """Checa terminação por sucesso para N mundos.

        Returns:
            terminated: [N] — bool
        """
        dist = np.linalg.norm(desired_goal - achieved_goal, axis=1)
        return dist < _THRESHOLD_POS

    # -----------------------------------------------------------------------
    # Utilitários
    # -----------------------------------------------------------------------

    @property
    def mjm(self) -> mujoco.MjModel:
        return self._mjm

    @property
    def m_gpu(self):
        return self._m_gpu

    @property
    def d_gpu(self):
        return self._d_gpu

    def close(self):
        pass
