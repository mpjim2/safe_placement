from gym.envs.registration import register

register(
    id='TactileObjectPlacementEnv-v0',
    entry_point='tactile_object_placement.envs.tactile_placing_env:TactileObjectPlacementEnv',
    max_episode_steps= 10000,
)