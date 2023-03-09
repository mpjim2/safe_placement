
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
    env.reset(options={'min_table_height' : 0.1, 'testing' :True})

    for x in range(5):
        
        env.reset(options={'min_table_height' : 0.1, 'testing' :True})
        # for _ in range(10):
        #     action = env.action_space.sample()
        #     env.step(action)
        time.sleep(30)  
        # time.sleep(0.5)
        # if done: break
    # env.reset()
        
# time.sleep(30)

    env.close()