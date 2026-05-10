import logging, os
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

def make_vec_envs(env_str, env_has_id, log_path, vec_env_cls, num_train, train_seed, eval_seed, rng_states_envs, override_monitor_logs, **env_kwargs):
    # adapted from stable_baselines3 (make_vec_env), but correctly sets env_id and rng_states
    # train_seed: seed over all train environments. The i-th environment seed wille be set with i+seed
    # eval_seed: seed over all test environments. The i-th environment seed wille be set with i+seed

    def make_env(env_id, seed, rng_states, monitor_fn_prefix):
        if env_has_id:
            env = gym.envs.make(env_str, **env_kwargs, env_id=env_id)
        else:
            env = gym.envs.make(env_str, **env_kwargs)

        if rng_states is None:
            # seed
            env_seed = seed + env_id
            env.action_space.seed(env_seed)
        else:
            env.set_rng_states(rng_states)

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