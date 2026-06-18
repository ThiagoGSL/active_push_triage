"""
warp_vec_env.py — Fase 3: VecEnv SB3-compatível usando MuJoCo Warp

Implementa a interface stable-baselines3 VecEnv sobre o
MuJoCoUR3PushSimpleWarpEnv (N mundos em batch GPU).

Uso no treinamento:
    from ur3_push_mujoco.gym_ur3_push.envs.mujoco.warp_vec_env import WarpVecEnv
    from ur3_push_mujoco.gym_ur3_push.envs.mujoco.ur3_push_simple_warp_env import MuJoCoUR3PushSimpleWarpEnv

    warp_env = MuJoCoUR3PushSimpleWarpEnv(nworld=64, n_substeps=40)
    vec_env  = WarpVecEnv(warp_env, max_episode_steps=200)
"""

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv


class WarpVecEnv(VecEnv):
    """VecEnv SB3-compatível sobre MuJoCoUR3PushSimpleWarpEnv.

    Adapta a interface de batch GPU do MJWarp para o formato que o SB3 espera:
    - obs como dicts com arrays [N, ...]
    - rewards como [N]
    - dones como [N]
    - infos como lista de N dicts

    Reseta automaticamente mundos que terminaram (done=True) ao início do
    próximo step, mantendo o pipeline de treinamento correto para episódios
    de comprimentos variáveis.

    Args:
        warp_env: instância de MuJoCoUR3PushSimpleWarpEnv já inicializada
        max_episode_steps: trunca episódios após este número de steps (TimeLimit)
    """

    def __init__(self, warp_env, max_episode_steps: int = 200):
        self._wenv = warp_env
        self.max_episode_steps = max_episode_steps
        self._elapsed = np.zeros(warp_env.nworld, dtype=np.int32)
        self._last_obs = None

        observation_space = spaces.Dict({
            "observation":   spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float64),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float64),
            "desired_goal":  spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float64),
        })
        action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        super().__init__(
            num_envs=warp_env.nworld,
            observation_space=observation_space,
            action_space=action_space,
        )

    # ------------------------------------------------------------------
    # Interface VecEnv obrigatória
    # ------------------------------------------------------------------

    def reset(self):
        """Reseta todos os N mundos e retorna observação inicial."""
        self._elapsed[:] = 0
        obs_dict = self._wenv.reset()
        self._last_obs = obs_dict
        return self._dict_to_array(obs_dict)

    def step_async(self, actions: np.ndarray):
        self._actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        """Executa o step e reseta mundos concluídos (done=True)."""
        obs_dict, rewards, terminateds, _, infos = self._wenv.step(self._actions)

        self._elapsed += 1
        truncateds = self._elapsed >= self.max_episode_steps

        dones = terminateds | truncateds

        # Informações de episódio para o SB3 logger
        for i in range(self._wenv.nworld):
            infos[i]["TimeLimit.truncated"] = bool(truncateds[i]) and not bool(terminateds[i])

        # Auto-reset de mundos concluídos
        done_ids = np.where(dones)[0]
        if len(done_ids) > 0:
            # Salva a obs terminal antes de resetar (SB3 precisa)
            obs_terminal = self._dict_to_array(obs_dict)
            for key in obs_dict:
                for i in done_ids:
                    infos[i]["terminal_observation"] = obs_terminal[i] if not isinstance(obs_terminal, dict) else {k: obs_terminal[k][i] for k in obs_terminal}

            reset_obs = self._wenv.reset(world_ids=done_ids)
            self._elapsed[done_ids] = 0

            # Substitui obs dos mundos resetados
            for key in obs_dict:
                obs_dict[key][done_ids] = reset_obs[key][done_ids]

        self._last_obs = obs_dict
        obs_array = self._dict_to_array(obs_dict)

        return obs_array, rewards.astype(np.float32), dones, infos

    def close(self):
        self._wenv.close()

    def seed(self, seed=None):
        pass  # seed gerenciado pelo MuJoCoUR3PushSimpleWarpEnv

    def get_attr(self, attr_name, indices=None):
        return [getattr(self._wenv, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self._wenv, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [getattr(self._wenv, method_name)(*method_args, **method_kwargs)] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    def _dict_to_array(self, obs_dict: dict):
        """Converte dict de obs batched para formato SB3 (dict de arrays [N,...])."""
        return obs_dict  # SB3 aceita dict de arrays diretamente com HerReplayBuffer

    def _reset_seeds(self):
        pass  # compatibilidade com make_envs.py
