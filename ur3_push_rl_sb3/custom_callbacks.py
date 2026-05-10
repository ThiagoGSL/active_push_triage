import os, shutil, pickle
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

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

    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> bool:
        self.n_calls_rollout_end += 1
        if self.n_calls_rollout_end >= self.calls_before_saving and self.n_calls_rollout_end % self.save_freq == 0:

            if self.verbose >= 2:
                print(f"Saving checkpoint to {self.save_path}")

            # delete old checkpoint 
            if os.path.exists(self.cp_path):
                shutil.rmtree(self.cp_path) 
            
            # RNG states of all Gymnasium environments
            rng_states_envs = {}
            # train envs
            rng_states_tmp = self.train_envs.get_attr("rng_states")
            for i in range(0, self.train_envs.num_envs):
                rng_states_envs.update({i: rng_states_tmp[i]})
            # eval env
            rng_states_tmp = self.eval_env.get_attr("rng_states")
            rng_states_envs.update({self.train_envs.num_envs: rng_states_tmp[0]})

            # save new checkpoint
            os.makedirs(self.cp_path, exist_ok=False)
            self.model.save_replay_buffer(path=os.path.join(self.cp_path,"replay_buffer")) # save replay buffer
            self.model.save(path=os.path.join(self.cp_path,"model")) # save last model 
            # save RNG states
            with open(os.path.join(self.cp_path,"rng_states_gymenvs.pkl"), mode="wb") as rng_env_file:
                pickle.dump(rng_states_envs, rng_env_file)
            # copy log and evaluation files 
            shutil.copytree(src=os.path.join(self.save_path,self.log_dir_name), dst=os.path.join(self.cp_path,self.log_dir_name))
            shutil.copytree(src=os.path.join(self.save_path,self.eval_dir_name), dst=os.path.join(self.cp_path,self.eval_dir_name))
            
        return True
