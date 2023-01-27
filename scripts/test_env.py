#!/home/marco/safe_placement/bin/python3
import sys
print(sys.path)
import gym
from gym_safe_placement.envs.gym_env import SafePlacementEnv

if __name__=='__main__':


    env = gym.make('SafePlacementEnv-v0')

    # env.set_object_params()