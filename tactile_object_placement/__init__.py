from gym.envs.registration import register

register(
    id='SafePlacementEnv-v0',
    entry_point='gym_safe_placement.envs.gym_env:SafePlacementEnv',
    max_episode_steps= 10000,
)