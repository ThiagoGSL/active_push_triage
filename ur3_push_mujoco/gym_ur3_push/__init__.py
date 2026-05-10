from gymnasium.envs.registration import register

def register_gymnasium_envs():
    #########################
    # MuJoCo                #
    #########################
    register(
        id='MujocoUR3PushSimpleEnv',
        entry_point='ur3_push_mujoco.gym_ur3_push.envs.mujoco.ur3_push_simple_env:MuJoCoUR3PushSimpleEnv',
        max_episode_steps=50
    )

    register(
        id='MujocoUR3PushEnv',
        entry_point='ur3_push_mujoco.gym_ur3_push.envs.mujoco.ur3_push_env:MuJoCoUR3PushEnv',
        max_episode_steps=50,
    )