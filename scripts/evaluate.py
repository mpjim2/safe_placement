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

State_reduced = namedtuple('State', 
                    ('state_myrmex',
                     'state_ep'))

State_reduced_2 = namedtuple('State', 
                    ('state_myrmex',
                     'state_jt'))

Transition = namedtuple('Transition',
                        ('state',
                         'action', 
                         'next_state',
                         'reward'))

def obs_to_input(obs, cur_stack, device, grid_size=16):

    cur_stack.state_myrmex.append(torch.cat([torch.from_numpy(obs['myrmex_r']).view(1, 1, 1,grid_size,grid_size),
                                             torch.from_numpy(obs['myrmex_l']).view(1, 1, 1,grid_size,grid_size)], 
                                             dim=1).to(device)) #.type(torch.DoubleTensor)
    
    cur_stack.state_ep.append(torch.from_numpy(obs["ee_pose"]).view(1, 1, 1, 6).to(device))                      #.type(torch.DoubleTensor)
    cur_stack.state_jp.append(torch.from_numpy(obs["joint_positions"]).view(1, 1, 1, 7).to(device))              #.type(torch.DoubleTensor)
    cur_stack.state_jt.append(torch.from_numpy(obs["joint_torques"]).view(1, 1, 1, 7).to(device))                #.type(torch.DoubleTensor)
    cur_stack.state_jv.append(torch.from_numpy(obs["joint_velocities"]).view(1, 1, 1, 7).to(device))             #.type(torch.DoubleTensor)
    
    return cur_stack

def NormalizeData(data, high, low):
    return (data - low) / (high - low)


def stack_to_state(state_stack, reduced=0):
    
    # for x in list(state_stack.state_myrmex):
    #     print(x.size())
     
    if reduced == 1:
        m = torch.cat(list(state_stack.state_myrmex), dim=2)
        ep = torch.cat(list(state_stack.state_ep), dim=2)
        return State_reduced(m, ep)
    if reduced == 2:
        m = torch.cat(list(state_stack.state_myrmex), dim=2)
        jt = torch.cat(list(state_stack.state_jt), dim=2)
        return State_reduced_2(m, jt)
    if not reduced:
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

def fingertip_hack(reading):

    ret = []
    s = 0
    for x in [4,4,4,4,2,2,2,2,2,2,2,2]:
        taxel = np.max(reading[s:s+x])
        s+=x
        ret.append(taxel)
    return np.array(ret)                                                                                                                                                                                                                                                    


def empty_state(reduced, device, tactile_shape, n_timesteps):
    if reduced == 0:
        state = State(state_myrmex=deque([torch.zeros(tactile_shape, dtype=torch.double, device=device) for _ in range(n_timesteps)], maxlen=n_timesteps),
                      state_ep=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=device) for _ in range(n_timesteps)], maxlen=n_timesteps),
                      state_jp=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=device) for _ in range(n_timesteps)], maxlen=n_timesteps),
                      state_jt=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=device) for _ in range(n_timesteps)], maxlen=n_timesteps),
                      state_jv=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=device) for _ in range(n_timesteps)], maxlen=n_timesteps))
    elif reduced == 1:
        state = State_reduced(state_myrmex=deque([torch.zeros(tactile_shape, dtype=torch.double, device=device) for _ in range(n_timesteps)], maxlen=n_timesteps),
                              state_ep=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=device) for _ in range(n_timesteps)], maxlen=n_timesteps))
    else:
        state = State_reduced_2(state_myrmex=deque([torch.zeros(tactile_shape, dtype=torch.double, device=device) for _ in range(n_timesteps)], maxlen=n_timesteps),
                                state_jt=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=device) for _ in range(n_timesteps)], maxlen=n_timesteps))
    
    return state
class DQN_Algo():

    def __init__(self, filepath, n_timesteps, reduced, architecture, sensor, grid_size=16, actionspace='reduced'):

        self.sensor = sensor
        self.FILEPATH = filepath 
        
        self.env = gym.make('TactileObjectPlacementEnv-v0', continuous=False, sensor=sensor, grid_size=grid_size, action_space=actionspace)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Policy and target network initilisation
        if architecture == 'temp_conv':
            if reduced == 1:
                self.policy_net = DQN.placenet_v2_reduced(n_actions=self.env.action_space.n, n_timesteps=n_timesteps, sensor_type=sensor, size=grid_size).double().to(self.device)
                self.target_net = DQN.placenet_v2_reduced(n_actions=self.env.action_space.n, n_timesteps=n_timesteps, sensor_type=sensor, size=grid_size).double().to(self.device)
                self.target_net.load_state_dict(self.policy_net.state_dict())
            elif reduced == 0:
                self.policy_net = DQN.dueling_placenet(n_actions=self.env.action_space.n, n_timesteps=n_timesteps, sensor_type=sensor, size=grid_size).double().to(self.device)
                self.target_net = DQN.dueling_placenet(n_actions=self.env.action_space.n, n_timesteps=n_timesteps, sensor_type=sensor, size=grid_size).double().to(self.device)
                self.target_net.load_state_dict(self.policy_net.state_dict())
            else: 
                self.policy_net = DQN.placenet_v3_reduced(n_actions=self.env.action_space.n, n_timesteps=n_timesteps, sensor_type=sensor, size=grid_size).double().to(self.device)
                self.target_net = DQN.placenet_v3_reduced(n_actions=self.env.action_space.n, n_timesteps=n_timesteps, sensor_type=sensor, size=grid_size).double().to(self.device)
                self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            if reduced:
                self.policy_net = DQN.placenet_LSTM_reduced(n_actions=self.env.action_space.n, n_timesteps=n_timesteps, sensor_type=sensor, size=grid_size).double().to(self.device)
                self.target_net = DQN.placenet_LSTM_reduced(n_actions=self.env.action_space.n, n_timesteps=n_timesteps, sensor_type=sensor, size=grid_size).double().to(self.device)
                self.target_net.load_state_dict(self.policy_net.state_dict()) 
            else:
                self.policy_net = DQN.placenet_LSTM(n_actions=self.env.action_space.n, n_timesteps=n_timesteps, sensor_type=sensor, size=grid_size).double().to(self.device)
                self.target_net = DQN.placenet_LSTM(n_actions=self.env.action_space.n, n_timesteps=n_timesteps, sensor_type=sensor, size=grid_size).double().to(self.device)
                self.target_net.load_state_dict(self.policy_net.state_dict()) 
            
        self.sensor=sensor
        if sensor == "fingertip":
            self.tactile_shape = (1,2,1,12)
        elif sensor == "plate":
            self.tactile_shape = (1,2,1,grid_size,grid_size)
            self.grid_size =grid_size
        
        
        self.obs_history = []
        #load NN Dictionary
        self.policy_net.load_state_dict(torch.load(self.FILEPATH + '/Policy_Model', map_location=self.device))
        # self.target_net.load_state_dict(torch.load(self.FILEPATH + '/Model'))
        self.reduced = reduced
        self.gapsize = 0.027
        self.angle_range = 0.17
        self.n_timesteps = n_timesteps
        self.cur_state_stack = State(state_myrmex=deque([torch.zeros((1,1,2,grid_size,grid_size), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_ep=deque([torch.zeros((1, 1, 1, 6), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_jp=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_jt=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_jv=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps))

    def _restart_env(self):
    
        self.env.close()
        self.env = gym.make('TactileObjectPlacementEnv-v0', continuous=False, sensor=self.sensor)

        return 0
    
    def _reset_env(self, options):

        restart_counter = 0
        success = False
        while not success:
            obs, info = self.env.reset(options=options)
            success = info['info']['success']
            
            if not success:
                print('Resetting Env unsuccessful. Restarting...')
                self._restart_env()
                restart_counter += 1
                if restart_counter >= 5:
                    self.env.close()

                    raise InterruptedError('Failed to reset Environment!')
                    break
        return obs, info

    def obs_to_input(self, obs, cur_stack, device):

        if self.sensor == 'plate':
            cur_stack.state_myrmex.append(torch.cat([torch.from_numpy(obs['myrmex_r']).view(1, 1, 1,self.grid_size,self.grid_size),
                                                    torch.from_numpy(obs['myrmex_l']).view(1, 1, 1,self.grid_size,self.grid_size)], 
                                                    dim=1).to(device)) #.type(torch.DoubleTensor)
        elif self.sensor == 'fingertip':
            right = fingertip_hack(obs['myrmex_r'])
            left  = fingertip_hack(obs['myrmex_l'])
            cur_stack.state_myrmex.append(torch.cat([torch.from_numpy(right).view(1, 1, 1, 12),
                                                     torch.from_numpy(left).view(1, 1, 1, 12)], 
                                                     dim=1).to(device)) #.type(torch.DoubleTensor)
        
                              #.type(torch.DoubleTensor)
        if self.reduced == 0 or self.reduced == 1:
            cur_stack.state_ep.append(torch.from_numpy(obs["ee_pose"]).view(1, 1, 1, 7).to(device))
        
        if self.reduced == 0:
            cur_stack.state_jp.append(torch.from_numpy(obs["joint_positions"]).view(1, 1, 1, 7).to(device))              #.type(torch.DoubleTensor)
            cur_stack.state_jt.append(torch.from_numpy(obs["joint_torques"]).view(1, 1, 1, 7).to(device))                #.type(torch.DoubleTensor)
            cur_stack.state_jv.append(torch.from_numpy(obs["joint_velocities"]).view(1, 1, 1, 7).to(device))             #.type(torch.DoubleTensor)

        if self.reduced == 2:
            cur_stack.state_jt.append(torch.from_numpy(obs["joint_torques"]).view(1, 1, 1, 7).to(device))                #.type(torch.DoubleTensor)

        return cur_stack

    def _normalize_observation(self, obs):
        
        normalized = {'observation' : {}}
        for key in obs['observation']:
            
            min_ = self.env.observation_space['observation'][key].high
            max_ = self.env.observation_space['observation'][key].low
            
            if key == 'ee_pose':
                min_ = min_[:3]
                max_ = max_[:3]

                pos  = NormalizeData(obs['observation'][key][:3], min_, max_)
                quat = obs['observation'][key][3:] 

                normalized['observation'][key] = np.concatenate([pos, quat])
            else:
                normalized['observation'][key] = NormalizeData(obs['observation'][key], min_, max_)
        return normalized
    
    def select_action(self, state, explore=True):
        
        
        action = self.policy_net(*state).max(1)[1].view(1,1)
        return action

    def test(self):
        
        obs, info = self._reset_env(options={'gap_size' : self.gapsize, 'testing' : False, 'angle_range' : self.angle_range, 'max_steps' : 200, 'sim_steps' : 50, 'reward_fn' : 'close_gap'})

        self.obs_history.append((obs, 0))
        obs = self._normalize_observation(obs)

        done = False

        #ReInitialize cur_state_stack
        self.cur_state_stack = self.cur_state_stack = empty_state(reduced=self.reduced, device=self.device, tactile_shape=self.tactile_shape, n_timesteps=self.n_timesteps)

        self.cur_state_stack = self.obs_to_input(obs["observation"], self.cur_state_stack, device=self.device)
        state = stack_to_state(self.cur_state_stack, reduced=self.reduced)

        for step in count():
            #experience sample: state, action, reward,  state+1
            action = self.select_action(state, explore=False)
            print(action)
            obs, reward, done, _ , info = self.env.step(action)
            if reward > 0:
                reward = 1
            self.obs_history.append((obs, reward))
            obs = self._normalize_observation(obs)
            
            reward = torch.tensor([reward])
            if not done:
                self.cur_state_stack = self.obs_to_input(obs["observation"], self.cur_state_stack, device=self.device)
                next_state = stack_to_state(self.cur_state_stack, reduced=self.reduced)
            else:
                next_state = None

            state = next_state

            if done:
                break

        print('Finished Evaluation! Reward: ', float(reward), " Steps until Done: ", step+1)
        
        with open(self.FILEPATH + '/obs_rewards_2.pickle', 'wb') as f:
            pickle.dump(self.obs_history, f)
        return reward, step+1

    
if __name__=='__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--savedir', required=False, help='Specify the directory where trained models should be saved')
    # #parser.add_argument('--continue_training', required=False, default='0')
    # opt = parser.parse_args()


    filepath = '/home/marco/Masterarbeit/Training/AllRuns/13-05-2023-17:07'
    algo = DQN_Algo(filepath=filepath,
                    n_timesteps=10, 
                    reduced=0,
                    architecture='temp_conv',
                    sensor='plate')

    algo.test()    
