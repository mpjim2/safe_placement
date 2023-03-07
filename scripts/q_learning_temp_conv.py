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

    def __init__(self, filepath, lr, expl_slope, discount_factor, mem_size, batch_size, n_epochs, tau, n_timesteps, global_step=None):

        self.FILEPATH = filepath 
        if not os.path.exists(self.FILEPATH):
            os.makedirs(self.FILEPATH)

        self.env = gym.make('TactileObjectPlacementEnv-v0')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Policy and target network initilisation
        self.policy_net = DQN.placenet_v2(n_actions=self.env.action_space.n, n_timesteps=n_timesteps).double().to(self.device)
        self.target_net = DQN.placenet_v2(n_actions=self.env.action_space.n, n_timesteps=n_timesteps).double().to(self.device)

        self.replay_buffer = ReplayMemory(mem_size)

        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = expl_slope
        
        self.discount_factor  = discount_factor
        self.BATCH_SIZE = batch_size
        self.N_EPOCHS = n_epochs

        self.LEARNING_RATE = lr

        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.LEARNING_RATE, amsgrad=True)

        self.soft_update_weight = tau
        
        if not global_step is None:
            self.stepcount=global_step
            #load NN Dictionary
            self.policy_net.load_state_dict(torch.load(self.FILEPATH + '/Model'))
            self.target_net.load_state_dict(torch.load(self.FILEPATH + '/Model'))

        else:
            self.stepcount = 0
        
        self.tableheight = 0.4

        self.n_timesteps = n_timesteps
        self.cur_state_stack = State(state_myrmex=deque([torch.zeros((1,1,2,16,16), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_ep=deque([torch.zeros((1, 1, 1, 6), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_jp=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_jt=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_jv=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps))


        #contains rewards & length of episode for every episode
        self.rewards_ = {'training' : [], 
                         'testing'  : [] }

        self.ep_lengths_ = {'training' : [],
                            'testing'  : []}

        self.done_causes = {'training' : [],
                           'testing'  : []}
    
    def save_checkpoint(self):
         
        torch.save(self.policy_net.state_dict(), self.FILEPATH + '/Model')
       
        with open(self.FILEPATH + '/Rewards.pickle', 'wb') as file:
            pickle.dump(self.rewards_, file)
        
        with open(self.FILEPATH + '/Ep_length.pickle', 'wb') as file:
            pickle.dump(self.ep_lengths_, file)
        
        with open(self.FILEPATH + '/done_causes.pickle', 'wb') as file:
            pickle.dump(self.done_causes, file)

        with open(self.FILEPATH + '/training_progress.pickle', 'wb') as file:
            progress = {'global_step' : self.stepcount}
            pickle.dump(progress, file)
        return 0
    
    def select_action(self, state):
        
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.stepcount / self.EPS_DECAY)
        if random.random() >= eps_threshold:
            action = self.policy_net(*state).max(1)[1].view(1,1)
        else:
            action = torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
        
        return action

    def test(self):
        
        obs, _ = self.env.reset()
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
            
            obs, reward, done, _ , info = self.env.step(action)

            reward = torch.tensor([reward])
            if not done:
                self.cur_state_stack = obs_to_input(obs["observation"], self.cur_state_stack, device=self.device)
                next_state = stack_to_state(self.cur_state_stack)
            else:
                next_state = None

            state = next_state

            if done:
                self.done_causes['testing'] = info['cause']
                break
        
        print('Finished Evaluation! Reward: ', float(reward), " Steps until Done: ", step)
        self.rewards_['testing'].append((float(reward), self.stepcount))
        self.ep_lengths_['testing'].append(step)
        
        return reward

    def train(self):
        
        for episode in range(1, self.N_EPOCHS+1):
        
            obs, info = self.env.reset()
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
                
                obs, reward, done, _ , info = self.env.step(action)

                reward = torch.tensor([reward])
                if not done:
                    self.cur_state_stack = obs_to_input(obs["observation"], self.cur_state_stack, device=self.device)
                    next_state = stack_to_state(self.cur_state_stack)
                else:
                    next_state = None

                self.replay_buffer.push(state, action, next_state, reward)

                state = next_state

                self.optimize()
                self.stepcount += 1

                if done:
                    self.done_causes['training'] = info['cause']
                    break
            
            print('Episode ', episode, ' done after ', step,  ' Steps ! reward: ', float(reward), ' Randomness: ', (self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.stepcount / self.EPS_DECAY)))
            
            self.rewards_['training'].append((float(reward), self.stepcount))
            self.ep_lengths_['training'].append(step)
            
            if episode % 5 == 0:
                r = self.test()
                if r > 0:
                    self.tableheight -= 0.1
                    if self.tableheight < 0.01:
                        self.tableheight = 0.01
                        
                self.save_checkpoint()


        self.save_checkpoint()
        self.env.close()

    def optimize(self):

        transitions = self.replay_buffer.sample(self.BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        state_batch = State(*zip(*batch.state))

        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                    batch.next_state)), device=self.device, dtype=torch.bool)

        
        state_action_values = self.policy_net(*(torch.cat(e) for e in state_batch)).gather(1, action_batch)

        #2. Compute Target Q-Values
        next_state_values = torch.zeros(action_batch.size()[0], device=self.device, dtype=torch.double)
        if any(non_final_mask):
            non_final_next_states = State(*zip(*[s for s in batch.next_state if s is not None]))
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(*(torch.cat(e) for e in non_final_next_states)).max(1)[0]

        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        self.update_target_network()

    def update_target_network(self):
        
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.soft_update_weight + target_net_state_dict[key]*(1-self.soft_update_weight)
        self.target_net.load_state_dict(target_net_state_dict)
    

    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', required=False, help='int', default='16')
    parser.add_argument('--nepochs', required=False, help='int', default='5')
    parser.add_argument('--lr', required=False, help='float', default="1e-4")
    parser.add_argument('--savedir', required=False, help='Specify the directory where trained models should be saved')
    #parser.add_argument('--continue_training', required=False, default='0')
    opt = parser.parse_args()


    batchsize = int(opt.batchsize)
    nepochs = int(opt.nepochs)
    lr = float(opt.lr)

    #continue_ = int(opt.continue_training)

    time_string = datetime.now().strftime("%d-%m-%Y-%H:%M")

        
    # if continue_:

    #     if opt.savedir is None:
    #         filepath = '/homes/mjimenezhaertel/Masterarbeit/Training/'
    #     else:
    #         filepath = opt.savedir 
    

    #     all_subdirs = [filepath+d for d in os.listdir(filepath) if os.path.isdir(filepath + d)]

    #     last_modified = max(all_subdirs, key=os.path.getmtime)

    #     with open(last_modified + '/training_progress.pickle', 'rb') as file:
    #         progress = pickle.load(file) 

    #     algo = DQN_Algo(filepath=last_modified,
    #                     lr=lr, 
    #                     expl_slope=15000, 
    #                     discount_factor=0.9, 
    #                     mem_size=7500, 
    #                     batch_size=batchsize, 
    #                     n_epochs=nepochs, 
    #                     tau=0.9,
    #                     n_timesteps=10, 
    #                     global_step=progress['global_step'])

    # else: 

    if opt.savedir is None:
        filepath = '/homes/mjimenezhaertel/Masterarbeit/Training/' + time_string + '/'
    else:
        filepath = opt.savedir + time_string + '/'

    algo = DQN_Algo(filepath=filepath,
                lr=lr, 
                expl_slope=10000, 
                discount_factor=0.9, 
                mem_size=5000, 
                batch_size=batchsize, 
                n_epochs=nepochs, 
                tau=0.9,
                n_timesteps=10)

    algo.train()    
