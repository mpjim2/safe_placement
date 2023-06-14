import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import DQN
import sys
import gym
import time
from tactile_object_placement.envs.tactile_placing_env import TactileObjectPlacementEnv_v2

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

State_reduced_3 = namedtuple('State', 
                    ('state_myrmex'))


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
    if reduced == 3:
        m = torch.cat(list(state_stack.state_myrmex), dim=2)
        return State_reduced_3(m)
    if not reduced:
        m = torch.cat(list(state_stack.state_myrmex), dim=2)
        ep = torch.cat(list(state_stack.state_ep), dim=2)
        jp = torch.cat(list(state_stack.state_jp), dim=2)
        jt = torch.cat(list(state_stack.state_jt), dim=2)
        jv = torch.cat(list(state_stack.state_jv), dim=2)

        return State(m, ep, jp, jt, jv)
        
    
def NormalizeData(data, high, low):
    return (data - low) / (high - low)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        
        if batch_size >= len(self.memory):
            batch_size = len(self.memory)   
            return random.sample(self.memory, batch_size)
        else: 
            return random.sample(self.memory, batch_size-1) + [self.memory[-1]]

    def __len__(self):
        return len(self.memory)

def fingertip_hack(reading):

    ret = []
    s = 0
    for x in [5,5,5,5]:
        taxel = np.max(reading[s:s+x])
        s+=x
        ret.append(taxel)

    
    return np.array(ret)                                                                                                                                                                                                                                                    

def approxRollingAvg(avg, sample, N):
    avg -= avg / N
    avg += sample / N

    return avg

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
    elif reduced == 2:
        state = State_reduced_2(state_myrmex=deque([torch.zeros(tactile_shape, dtype=torch.double, device=device) for _ in range(n_timesteps)], maxlen=n_timesteps),
                                state_jt=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=device) for _ in range(n_timesteps)], maxlen=n_timesteps))
    else: 
        state = State_reduced_3(state_myrmex=deque([torch.zeros(tactile_shape, dtype=torch.double, device=device) for _ in range(n_timesteps)], maxlen=n_timesteps))
    return state

class DQN_Algo():

    def __init__(self, filepath, sensor="plate", actionspace='full', grid_size=16, timesteps=20):

        self.env = gym.make('TactileObjectPlacementEnv-v1', continuous=False, sensor=sensor, grid_size=grid_size, action_space=actionspace, timesteps=timesteps)
        

        if not os.path.exists(filepath):
            os.makedirs(filepath)

        self.filepath = filepath

        self.max_ep_steps = 20

        self.timesteps = timesteps
        self.gapsize = 0.002
        self.angle_range = 0.0
        self.sim_steps = 10

        self.train_avg = 0
        self.test_avg = 0

       
        if actionspace == 'full':

            self.reward_fn = 'close_gap'
        else:
            self.reward_fn = 'place'
            
        if sensor == "fingertip":
            self.tactile_shape = (1,2,1,12)
        elif sensor == "plate":
            self.tactile_shape = (1,2,1,grid_size,grid_size)
        
        #contains rewards & length of episode for every episode

    def obs_to_input(self, obs, cur_stack, device):

        if self.sensor == 'plate':
            cur_stack.state_myrmex.append(torch.cat([torch.from_numpy(obs['myrmex_r']).view(1, 1, 1,self.grid_size,self.grid_size),
                                                    torch.from_numpy(obs['myrmex_l']).view(1, 1, 1,self.grid_size,self.grid_size)], 
                                                    dim=1).to(device)) #.type(torch.DoubleTensor)
        elif self.sensor == 'fingertip':
            right = fingertip_hack(obs['myrmex_r'])
            
            print(right)
            
            self.tactile_test = right
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

    def test(self, action=0, save=0):
        
        obs, info = self._reset_env(options={'gap_size' : self.gapsize, 'testing' : True, 'angle_range' : self.angle_range, 'max_steps' : self.max_ep_steps, 'sim_steps' : self.sim_steps, 'reward_fn' :self.reward_fn})

        done = False
        
        RIGHT = []
        LEFT = []
        contact = None
        
        cumulative_reward = 0
        for step in count():
            #experience sample: state, action, reward,  state+1
            print(step)
            #action = torch.tensor([[self.env.action_space.n - 1]], dtype=torch.long)

            obs, reward, done, _ , info = self.env.step(action)
            
            L = obs['observation']['myrmex_l']
            R = obs['observation']['myrmex_r']

            for l in L:
                LEFT.append(l)

            for r in R:
                RIGHT.append(r)
            # if obs['observation']['contact'] and contact is None: 
            #     contact = step*self.timesteps
            
            cumulative_reward += reward
            reward = torch.tensor([reward])

            if done or step >= self.max_ep_steps:
                break           
            
        # print('Contact at ', contact)

        data = (LEFT, RIGHT, contact)
        if save == 1:
            with open(self.filepath + 'data_static.pickle', 'wb') as f:
                pickle.dump(data, f)
        else:
            with open(self.filepath + 'data.pickle', 'wb') as f:
                pickle.dump(data, f)

        return reward, cumulative_reward, step+1
    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', required=False, help='int', default='16')
    parser.add_argument('--nepochs', required=False, help='int', default='5')
    parser.add_argument('--lr', required=False, help='float', default="1e-4")
    parser.add_argument('--savedir', required=False, help='Specify the directory where trained models should be saved')
    parser.add_argument('--mem_size', required=False, default='7500')
    parser.add_argument('--expl_slope', required=False, default='50000')
    parser.add_argument('--sensor', required=False, default='plate')
    parser.add_argument('--architecture', required=False, default='temp_conv')
    parser.add_argument('--reduced_state', required=False, default='0')
    parser.add_argument('--continue_', required=False, default='0')
    parser.add_argument('--global_step', required=False, default='0')
    parser.add_argument('--episode', required=False, default='0')    
    parser.add_argument('--actionspace', required=False, default='full')
    opt = parser.parse_args()

    sensor = opt.sensor
    expl_slope = int(opt.expl_slope)
    mem_size = int(opt.mem_size)
    batchsize = int(opt.batchsize)
    nepochs = int(opt.nepochs)
    lr = float(opt.lr)
    reduced = int(opt.reduced_state)
    global_step = int(opt.global_step)
    episode = int(opt.episode)
    actionspace = opt.actionspace
    continue_ = bool(int(opt.continue_))

    if not continue_:
        time_string = datetime.now().strftime("%d-%m-%Y-%H:%M")
        global_step = None
        if opt.savedir is None:
            filepath = '/homes/mjimenezhaertel/Masterarbeit/Training/' + time_string + '/'
        else:
            filepath = opt.savedir + time_string + '/'
        
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        with open(filepath + 'options.txt', 'w') as f:
            f.write("sensor: " + sensor + "\n")
            f.write("expl_slope: " + opt.expl_slope + "\n")
            f.write("mem_size: " + opt.mem_size + "\n")
            f.write("batchsize: " + opt.batchsize + "\n")
            f.write("nepochs: " + opt.nepochs + "\n")
            f.write("lr: " + opt.lr + "\n")
            f.write("reduced: " + opt.reduced_state + "\n")
            f.write("architecture: " + opt.architecture)
    else:
        filepath = opt.savedir   
        


    algo = DQN_Algo(filepath=filepath,
                    lr=lr, 
                    expl_slope=expl_slope, 
                    discount_factor=0.9, 
                    mem_size=mem_size, 
                    batch_size=batchsize, 
                    n_epochs=nepochs, 
                    tau=0.95,
                    n_timesteps=5,
                    sensor=sensor,
                    global_step=global_step,
                    architecture=opt.architecture,
                    reduced=reduced,
                    episode = episode,
                    actionspace=actionspace)

    algo.train()    
    algo.summary_writer.close()
