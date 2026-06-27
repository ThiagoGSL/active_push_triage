import os, shutil, pickle
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env import VecNormalize

class CustomCheckpointCallback(BaseCallback):

    def __init__(self,
                calls_before_saving: int,
                save_freq: int,
                save_path: str,
                cp_path: str,
                train_envs: VecEnv,
                eval_env: VecEnv,
                log_dir_name:str = "logs",
                eval_dir_name: str = "evaluation",
                verbose: int = 0):
        
        super().__init__(verbose)
        self.calls_before_saving = calls_before_saving
        self.save_freq = save_freq
        self.save_path = save_path
        self.cp_path = cp_path
        self.log_dir_name = log_dir_name
        self.eval_dir_name = eval_dir_name
        self.train_envs = train_envs
        self.eval_env = eval_env
        self.n_calls_rollout_end = 0
        self._next_sac_save = None  # set lazily on first step

    def _on_step(self) -> bool:
        # Off-policy algorithms (SAC) never call _on_rollout_end.
        # Fix: SAC should save based on a large timestep threshold (e.g. 500k steps) 
        # because save_freq is calibrated for PPO rollouts.
        is_on_policy = hasattr(self.model, 'n_steps')  # PPO has n_steps; SAC does not
        if not is_on_policy:
            if self._next_sac_save is None:
                self._next_sac_save = 500_000  # Save every 500k timesteps
            if self.model.num_timesteps >= self._next_sac_save:
                self._save_checkpoint()
                self._next_sac_save += 500_000  # advance to next threshold
        return True
    
    def _on_rollout_end(self) -> bool:
        # On-policy algorithms (PPO): save after N rollouts
        is_on_policy = hasattr(self.model, 'n_steps')
        if is_on_policy:
            self.n_calls_rollout_end += 1
            if self.n_calls_rollout_end >= self.calls_before_saving and self.n_calls_rollout_end % self.save_freq == 0:
                self._save_checkpoint()
        return True

    def _save_checkpoint(self):
        if self.verbose >= 2:
            print(f"Saving checkpoint to {self.save_path}")

        # delete old checkpoint 
        if os.path.exists(self.cp_path):
            shutil.rmtree(self.cp_path) 
        
        # RNG states of all Gymnasium environments
        # (opcional: WarpVecEnv nao expoe rng_states por mundo)
        rng_states_envs = {}
        try:
            # train envs
            rng_states_tmp = self.train_envs.get_attr("rng_states")
            for i in range(0, self.train_envs.num_envs):
                rng_states_envs.update({i: rng_states_tmp[i]})
            # eval env
            rng_states_tmp = self.eval_env.get_attr("rng_states")
            rng_states_envs.update({self.train_envs.num_envs: rng_states_tmp[0]})
        except (AttributeError, NotImplementedError):
            pass  # WarpVecEnv: sem rng_states por mundo (GPU nao tem RNG sequencial)

        # save new checkpoint
        os.makedirs(self.cp_path, exist_ok=False)
        self.model.save(path=os.path.join(self.cp_path, "model"))  # save model
        # save RNG states
        with open(os.path.join(self.cp_path, "rng_states_gymenvs.pkl"), mode="wb") as rng_env_file:
            pickle.dump(rng_states_envs, rng_env_file)
        # save VecNormalize stats (if applicable)
        env = self.model.get_vec_normalize_env()
        if env is not None:
            vecnorm_path = os.path.join(self.cp_path, 'vecnormalize.pkl')
            env.save(vecnorm_path)
            if self.verbose >= 2:
                print(f"[VecNormalize] Stats salvas em: {vecnorm_path}")
        # copy log and evaluation files
        shutil.copytree(src=os.path.join(self.save_path, self.log_dir_name), dst=os.path.join(self.cp_path, self.log_dir_name))
        shutil.copytree(src=os.path.join(self.save_path, self.eval_dir_name), dst=os.path.join(self.cp_path, self.eval_dir_name))
