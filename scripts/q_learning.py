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


State = namedtuple('State', 
                    ('state_ml',
                     'state_mr', 
                     'state_ep', 
                     'state_jp', 
                     'state_jt', 
                     'state_jv'))

Transition = namedtuple('Transition',
                        ('state',
                         'action', 
                         'next_state',
                         'reward'))

def obs_to_input(obs):

    myrmex_l = torch.from_numpy(obs["myrmex_l"].reshape((1, 16,16))).unsqueeze(0) #.type(torch.DoubleTensor)
    myrmex_r = torch.from_numpy(obs["myrmex_r"].reshape((1, 16,16))).unsqueeze(0) #.type(torch.DoubleTensor)
    pose     = torch.from_numpy(obs["ee_pose"]).unsqueeze(0)                      #.type(torch.DoubleTensor)
    j_pos    = torch.from_numpy(obs["joint_positions"]).unsqueeze(0)              #.type(torch.DoubleTensor)
    j_tau    = torch.from_numpy(obs["joint_torques"]).unsqueeze(0)                #.type(torch.DoubleTensor)
    j_vel    = torch.from_numpy(obs["joint_velocities"]).unsqueeze(0)             #.type(torch.DoubleTensor)
    
    return State(myrmex_l, myrmex_r, pose, j_pos, j_tau, j_vel)

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

    def __init__(self, lr, expl_slope, discount_factor, mem_size, batch_size, n_epochs, tau):

        self.env = TactileObjectPlacementEnv()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Policy and target network initilisation
        self.policy_net = DQN.place_net().double()
        self.target_net = DQN.place_net().double()

        self.replay_buffer = ReplayMemory(mem_size)

        self.exploration_rate = 0.9
        self.discount_factor  = discount_factor
        self.BATCH_SIZE = batch_size
        self.N_EPOCHS = n_epochs

        self.LEARNING_RATE = lr

        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.LEARNING_RATE, amsgrad=True)

        self.soft_update_weight = tau

    def train(self):
        
        for episode in range(self.N_EPOCHS):
        
            obs, info = self.env.reset()
            done = False

            state = obs_to_input(obs["observation"])

            while not done:
                #experience sample: state, action, reward,  state+1
                            
                if np.random.uniform(0, 1) >= self.exploration_rate:
                    action = self.policy_net(*state).max(1)[1].view(1,1)

                else:
                    action = torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
                
                obs, reward, done, _ , _ = self.env.step(action)
                
                reward = torch.tensor([reward])
                if not done:
                    next_state = obs_to_input(obs["observation"])
                else:
                    next_state = None

                self.replay_buffer.push(state, action, next_state, reward)

                state = next_state

                self.optimize()

        self.enc.close()
    
    def optimize(self):
        transitions = self.replay_buffer.sample(self.BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        state_batch = State(*zip(*batch.state))

        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                    batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = State(*zip(*[s for s in batch.next_state if s is not None]))

        state_action_values = self.policy_net(*(torch.cat(e) for e in state_batch)).gather(1, action_batch)

        #2. Compute Target Q-Values
        next_state_values = torch.zeros(action_batch.size()[0], device=self.device, dtype=torch.double)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(*(torch.cat(e) for e in non_final_next_states)).max(1)[0]

        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        # non_final_next_states = torch.cat([s for s in batch.next_state_ml
        #                                             if s is not None])

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

    algo = DQN_Algo(lr=1e-4, 
                    expl_slope=10000, 
                    discount_factor=0.9, 
                    mem_size=1000, 
                    batch_size=4, 
                    n_epochs=2, 
                    tau=0.9)

    algo.train()