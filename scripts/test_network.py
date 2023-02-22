import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import DQN
import sys
import gym
import time
from tactile_object_placement.envs.tactile_placing_env import TactileObjectPlacementEnv

import numpy as np


def obs_to_input(obs):

    myrmex_l = torch.from_numpy(obs["myrmex_l"].reshape((1, 16,16))) #.type(torch.DoubleTensor)
    myrmex_r = torch.from_numpy(obs["myrmex_r"].reshape((1, 16,16))) #.type(torch.DoubleTensor)
    pose     = torch.from_numpy(obs["ee_pose"])                      #.type(torch.DoubleTensor)
    j_pos    = torch.from_numpy(obs["joint_positions"])              #.type(torch.DoubleTensor)
    j_tau    = torch.from_numpy(obs["joint_torques"])                #.type(torch.DoubleTensor)
    j_vel    = torch.from_numpy(obs["joint_velocities"])             #.type(torch.DoubleTensor)
    
    return myrmex_r, myrmex_l, pose, j_pos, j_tau, j_vel

def output_to_action(output):

    action = {"move_down" : 0, 
              "rotate_X" : 0,
              "rotate_Y" : 0,
              "open_gripper" : 0}
    
    if output[0].detach().numpy() >= 0.5:
        action["open_gripper"] = 1
    
    action["rotate_X"]  = np.argmax(output[1].detach().numpy()) -1
    action["rotate_Y"]  = np.argmax(output[2].detach().numpy()) -1
    
    if output[3].detach().numpy()  >= 0.5:
        action["move_down"] = -1

    return action

if __name__=='__main__':

    place_net = DQN.place_net().double()
    env = gym.make('TactileObjectPlacementEnv-v0')

    obs, info = env.reset()
    done = False

    counter = 0

    while not done:
        m1, m2, p, j_p, j_t, j_v = obs_to_input(obs["observation"])
        output = place_net(m1, m2, p, j_p, j_t, j_v)
        action = output_to_action(output)
        obs, reward, done, _ , _ = env.step(action)

        counter += 1
        print(action["open_gripper"], counter)
        if counter == 100:
            done = True

    env.close()
        
