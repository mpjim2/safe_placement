
#!/usr/bin/env python3
import sys
import gym
import time
from tactile_object_placement.envs.tactile_placing_env import TactileObjectPlacementEnv
import numpy as np

if __name__=="__main__":

    env = gym.make('TactileObjectPlacementEnv-v0', sensor="fingertip", continuous=False)

    # env.reset_world()
    # for i in range(10):

    # env.reset()    


    # env.reset()
    # env.reset(options={'min_table_height' : 0.1, 'testing' :True})
    case_found = False
    n = env.action_space.n

    g = 0.002
    ar = 0.17

    for action in range(10000):
        
        print('########################')
        print(action)
        print('########################')

        env.reset(options={'gap_size' : 0.002, 'angle_range' : 0.17, 'testing' : False})
        done = False
        time.sleep(0.5)
        if action %100 == 0:
            g += 0.002
            ar += 0.05
        
        # time.sleep(3)
        # while not done:
        #     _, r, done, _, _ = env.step(3)
            
        #     if r == -0.5:
        #         case_found = True
        #         time.sleep(10)
        # # if done: break
    # env.reset()
        
# time.sleep(30)

    env.close()