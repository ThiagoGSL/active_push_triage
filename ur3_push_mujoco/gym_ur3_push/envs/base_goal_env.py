import numpy as np
import gymnasium as gym

class BaseGoalEnv(gym.Env):

    def __init__(self):
        self.rng_noise = np.random.default_rng(seed=None)

    def reset(self, seed=None, options={}):
        super().reset(seed=seed)
        if seed is not None:
            self.rng_noise = np.random.default_rng(seed=seed)
    
    def step(self, action):
        raise NotImplementedError

    def close(self):
        pass

    def compute_terminated(self, achieved_goal, desired_goal, info={}):
        """The objective is to reach the goal for an indefinite period of time. (gymnasium_robotics -> FetchPushEnv)"""
        return False

    def compute_truncated(self, achieved_goal, desired_goal, info={}):
        """The environments will be truncated only if setting a time limit with max_steps which will automatically wrap the environment in a gymnasium TimeLimit wrapper."""
        return False

    def compute_reward(self, achieved_goal, desired_goal, info={}):
        raise NotImplementedError

    @property
    def rng_states(self):
        return {"gym_rng": self.np_random.__getstate__(),
                "noise_rng": self.rng_noise.__getstate__(),
                "action_space": self.action_space.np_random.__getstate__()}
    
    def set_rng_states(self, rng_states):
        self.np_random.__setstate__(rng_states["gym_rng"])
        self.rng_noise.__setstate__(rng_states["noise_rng"])
        self.action_space.np_random.__setstate__(rng_states["action_space"])

    def _get_obs(self):
        raise NotImplementedError

    def _get_info(self):
        raise NotImplementedError
