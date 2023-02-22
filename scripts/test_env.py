
#!/usr/bin/env python3
import sys
import gym
import time
from tactile_object_placement.envs.tactile_placing_env import TactileObjectPlacementEnv
import numpy as np

if __name__=="__main__":

    env = gym.make('TactileObjectPlacementEnv-v0')

    # env.reset_world()
    # for i in range(10):

    # env.reset()    


    # env.reset()
    env.reset()

    for x in range(10):
        
        env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            env.step(action)
        time.sleep(2)  
        # time.sleep(0.5)
        # if done: break
    # env.reset()
    time.sleep(1)
# time.sleep(30)

    env.close()