import logging, os
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor


def make_warp_env(num_train: int,
                  max_episode_steps: int = 200,
                  sparse_reward: bool = True,
                  ee_to_obj_reward_scale: float = 0.2,
                  action_scaling_factor: float = 0.6,
                  n_substeps: int = 40,
                  seed: int = 0):
    """Cria WarpVecEnv (MuJoCo Warp, GPU batch) compatível com SB3.

    Substitui make_vec_envs() quando --useWarp=1 é passado.
    Retorna (train_envs, eval_env) para compatibilidade com o script train.py.

    Args:
        num_train: número de mundos em paralelo na GPU (ex: 64, 128, 256)
        max_episode_steps: truncamento de episódios (TimeLimit)
        sparse_reward: True = recompensa esparsa (-1/0)
        ee_to_obj_reward_scale: escala do reward de aproximação EE→obj
        action_scaling_factor: fator de escala das ações
        n_substeps: substeps de física por gym step
        seed: seed do RNG

    Returns:
        train_envs: WarpVecEnv com num_train mundos
        eval_env: WarpVecEnv com 1 mundo (para avaliação)
    """
    from ur3_push_mujoco.gym_ur3_push.envs.mujoco.ur3_push_simple_warp_env import MuJoCoUR3PushSimpleWarpEnv
    from ur3_push_mujoco.gym_ur3_push.envs.mujoco.warp_vec_env import WarpVecEnv

    # Ambiente de treinamento: num_train mundos
    warp_train = MuJoCoUR3PushSimpleWarpEnv(
        nworld=num_train,
        n_substeps=n_substeps,
        sparse_reward=sparse_reward,
        ee_to_obj_reward_scale=ee_to_obj_reward_scale,
        action_scaling_factor=action_scaling_factor,
        seed=seed,
    )
    train_envs = WarpVecEnv(warp_train, max_episode_steps=max_episode_steps)

    # Ambiente de avaliação: 1 mundo (mesmo config, seed diferente)
    warp_eval = MuJoCoUR3PushSimpleWarpEnv(
        nworld=1,
        n_substeps=n_substeps,
        sparse_reward=sparse_reward,
        ee_to_obj_reward_scale=ee_to_obj_reward_scale,
        action_scaling_factor=action_scaling_factor,
        seed=seed + 99999,
    )
    eval_env = WarpVecEnv(warp_eval, max_episode_steps=max_episode_steps)

    return train_envs, eval_env

def make_vec_envs(env_str, env_has_id, log_path, vec_env_cls, num_train, train_seed, eval_seed, rng_states_envs, override_monitor_logs, max_episode_steps=None, **env_kwargs):
    # adapted from stable_baselines3 (make_vec_env), but correctly sets env_id and rng_states
    # train_seed: seed over all train environments. The i-th environment seed wille be set with i+seed
    # eval_seed: seed over all test environments. The i-th environment seed wille be set with i+seed

    def make_env(env_id, seed, rng_states, monitor_fn_prefix):
        if env_has_id:
            env = gym.envs.make(env_str, **env_kwargs, env_id=env_id)
        else:
            env = gym.envs.make(env_str, **env_kwargs)

        # Seed / restore RNG state before any wrapper is applied
        if rng_states is None:
            env_seed = seed + env_id
            env.action_space.seed(env_seed)
        else:
            env.set_rng_states(rng_states)

        if max_episode_steps is not None and max_episode_steps > 0:
            env = TimeLimit(env, max_episode_steps=max_episode_steps)

        # Monitor wrapper
        os.makedirs(log_path, exist_ok=True)
        env = Monitor(env, 
                      filename=os.path.join(log_path, f"{monitor_fn_prefix}_env_{env_id}"), info_keywords=("is_success",), 
                      override_existing=override_monitor_logs)

        return env

    train_envs = None
    eval_env = None 

    if "env_id" in env_kwargs.keys():
        logging.error("env_id cannot be set manually when make_subprocvec_envs() is called")
        return train_envs, eval_env
    
    if num_train > 0:
        train_envs = vec_env_cls([lambda x=i: make_env( env_id=x, 
                                                        seed=train_seed, 
                                                        rng_states=rng_states_envs[x] if rng_states_envs is not None else None, 
                                                        monitor_fn_prefix="train") 
                                                        for i in range(0, num_train)])
    
    # eval env
    eval_env = vec_env_cls([lambda: make_env(env_id=num_train, 
                                            seed=eval_seed, 
                                            rng_states=rng_states_envs[list(rng_states_envs.keys())[-1]] if rng_states_envs is not None else None, 
                                            monitor_fn_prefix="eval")])
     
    # Prepare the seeds for the first reset
    if rng_states_envs is None:
        if train_envs is not None:
            train_envs.seed(train_seed)
        eval_env.seed(eval_seed)
    else:
        if train_envs is not None:
            train_envs._reset_seeds()
        eval_env._reset_seeds()

    return train_envs, eval_env