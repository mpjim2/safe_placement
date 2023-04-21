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

    for i in range(1):
   
        env.reset(options={'gap_size' : 0.1, 'angle_range' : 0.17, 'testing' : False, 'sim_steps' : 10, 'max_steps' : 500})
        done = False

        g += 0.005

        while not done:
            obs, r, done, _, info = env.step(3)

            # print(obs['observation']['myrmex_l'])
            if r > 0:
                time.sleep(2)
                print(r)
                break
        # if done: break
    # env.reset()
        
# time.sleep(30)

    env.close()