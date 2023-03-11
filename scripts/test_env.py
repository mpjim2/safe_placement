
#!/usr/bin/env python3
import sys
import gym
import time
from tactile_object_placement.envs.tactile_placing_env import TactileObjectPlacementEnv
import numpy as np

if __name__=="__main__":

    env = gym.make('TactileObjectPlacementEnv-v0', sensor="plate", continuous=False)

    # env.reset_world()
    # for i in range(10):

    # env.reset()    


    # env.reset()
    # env.reset(options={'min_table_height' : 0.1, 'testing' :True})

    for x in range(1):
        env.reset(options={'min_table_height' : 0.1, 'testing' :True})
        for _ in range(50):
            #action = env.action_space.sample()
            env.step(action=0)
            time.sleep(2)
        # if done: break
    # env.reset()
        
# time.sleep(30)

    env.close()