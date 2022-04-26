from gym.envs.registration import register

register(
    id='WildlifeEnv-v0',
    entry_point='marlenvs.envs:WildlifeEnv',
)

register(
    id='JunctionEnv-v0',
    entry_point='marlenvs.envs:JunctionEnv',
)
