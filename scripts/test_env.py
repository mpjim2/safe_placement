
#!/usr/bin/env python3
import sys
import gym
import time
from gym_safe_placement.envs.gym_env import SafePlacementEnv



if __name__=="__main__":

    env = gym.make('SafePlacementEnv-v0')

    # env.reset_world()
    # for i in range(10):

    # env.reset()    

    for i in range(1):
        print("Attempt: ", i)   
        env.reset()
        for x in range(20):
            action = env.action_space.sample()
            action['open_gripper'] = 0
            obs, _, done, _, _ = env.step(action)
            
            # print(env.observation_space['observation']['joint_velocities'].dtype, obs['observation']['joint_velocities'].dtype)
            time.sleep(0.5)
            if done: break
        # env.reset()
        time.sleep(1)
    # time.sleep(30)

    env.close()