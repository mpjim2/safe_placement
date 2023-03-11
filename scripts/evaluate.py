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
from collections import deque, namedtuple
import random
import math
from itertools import count

import os
import pickle
from datetime import datetime

import glob
import argparse

State = namedtuple('State', 
                    ('state_myrmex',
                     'state_ep', 
                     'state_jp', 
                     'state_jt', 
                     'state_jv'))

Transition = namedtuple('Transition',
                        ('state',
                         'action', 
                         'next_state',
                         'reward'))

def obs_to_input(obs, cur_stack, device):

    cur_stack.state_myrmex.append(torch.cat([torch.from_numpy(obs['myrmex_r']).view(1, 1, 1,16,16),
                                             torch.from_numpy(obs['myrmex_l']).view(1, 1, 1,16,16)], 
                                             dim=1).to(device)) #.type(torch.DoubleTensor)
    
    cur_stack.state_ep.append(torch.from_numpy(obs["ee_pose"]).view(1, 1, 1, 6).to(device))                      #.type(torch.DoubleTensor)
    cur_stack.state_jp.append(torch.from_numpy(obs["joint_positions"]).view(1, 1, 1, 7).to(device))              #.type(torch.DoubleTensor)
    cur_stack.state_jt.append(torch.from_numpy(obs["joint_torques"]).view(1, 1, 1, 7).to(device))                #.type(torch.DoubleTensor)
    cur_stack.state_jv.append(torch.from_numpy(obs["joint_velocities"]).view(1, 1, 1, 7).to(device))             #.type(torch.DoubleTensor)
    
    return cur_stack

def stack_to_state(state_stack):
    
    m = torch.cat(list(state_stack.state_myrmex), dim=2) 
    ep = torch.cat(list(state_stack.state_ep), dim=2)
    jp = torch.cat(list(state_stack.state_jp), dim=2)
    jt = torch.cat(list(state_stack.state_jt), dim=2)
    jv = torch.cat(list(state_stack.state_jv), dim=2)

    return State(m, ep, jp, jt, jv)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            batch_size = len(self.memory)   
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN_Algo():

    def __init__(self, filepath, n_timesteps):

        self.FILEPATH = filepath 
        if not os.path.exists(self.FILEPATH):
            os.makedirs(self.FILEPATH)

        self.env = gym.make('TactileObjectPlacementEnv-v0')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Policy and target network initilisation
        self.policy_net = DQN.placenet_v2(n_actions=self.env.action_space.n, n_timesteps=n_timesteps).double().to(self.device)
        self.target_net = DQN.placenet_v2(n_actions=self.env.action_space.n, n_timesteps=n_timesteps).double().to(self.device)


        #load NN Dictionary
        self.policy_net.load_state_dict(torch.load(self.FILEPATH + '/Model', map_location=torch.device('cpu')))
        # self.target_net.load_state_dict(torch.load(self.FILEPATH + '/Model'))


        self.tableheight = 0.4

        self.n_timesteps = n_timesteps
        self.cur_state_stack = State(state_myrmex=deque([torch.zeros((1,1,2,16,16), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_ep=deque([torch.zeros((1, 1, 1, 6), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_jp=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_jt=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_jv=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps))



    
    def select_action(self, state):
        
        action = self.policy_net(*state).max(1)[1].view(1,1)
     
        return action

    def test(self):
        
        obs, info = self.env.reset(options={'min_table_height' : self.tableheight, 'testing' : True})

        done = False

        #ReInitialize cur_state_stack
        self.cur_state_stack = State(state_myrmex=deque([torch.zeros((1,2,1, 16,16), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_ep=deque([torch.zeros((1, 1, 1, 6), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_jp=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_jt=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_jv=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps))

        self.cur_state_stack = obs_to_input(obs["observation"], self.cur_state_stack, device=self.device)
        state = stack_to_state(self.cur_state_stack)

        for step in count():
            #experience sample: state, action, reward,  state+1
            action = self.select_action(state)
            
            print(self.cur_state_stack.state_ep[0])
            obs, reward, done, _ , info = self.env.step(action)

            reward = torch.tensor([reward])
            if not done:
                self.cur_state_stack = obs_to_input(obs["observation"], self.cur_state_stack, device=self.device)
                next_state = stack_to_state(self.cur_state_stack)
            else:
                next_state = None

            state = next_state

            if done:
                break

        
        return reward

    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', required=False, help='Specify the directory where trained models should be saved')
    #parser.add_argument('--continue_training', required=False, default='0')
    opt = parser.parse_args()

    filepath = opt.savedir
    algo = DQN_Algo(filepath=filepath,
                    n_timesteps=10)

    algo.test()    
