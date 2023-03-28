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



def stack_to_state(state_stack):
    
    # for x in list(state_stack.state_myrmex):
    #     print(x.size())
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
    for x in [4,4,4,4,2,2,2,2,2,2,2,2]:
        taxel = np.max(reading[s:s+x])
        s+=x
        ret.append(taxel)
    return np.array(ret)                                                                                                                                                                                                                                                    

class DQN_Algo():

    def __init__(self, filepath, lr, expl_slope, discount_factor, mem_size, batch_size, n_epochs, tau, n_timesteps, sensor="plate", global_step=None):

        self.sensor = sensor
        self.FILEPATH = filepath 
        if not os.path.exists(self.FILEPATH):
            os.makedirs(self.FILEPATH)

        self.env = gym.make('TactileObjectPlacementEnv-v0', continuous=False, sensor=sensor)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Policy and target network initilisation
        self.policy_net = DQN.placenet_v2(n_actions=self.env.action_space.n, n_timesteps=n_timesteps, sensor_type=sensor).double().to(self.device)
        self.target_net = DQN.placenet_v2(n_actions=self.env.action_space.n, n_timesteps=n_timesteps, sensor_type=sensor).double().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.replay_buffer = ReplayMemory(mem_size)

        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = expl_slope
        
        self.discount_factor  = discount_factor
        self.BATCH_SIZE = batch_size
        self.N_EPOCHS = n_epochs

        self.LEARNING_RATE = lr

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.LEARNING_RATE, amsgrad=True)

        self.soft_update_weight = tau
        
        if not global_step is None:
            self.stepcount=global_step
            #load NN Dictionary
            self.policy_net.load_state_dict(torch.load(self.FILEPATH + '/Model'))
            self.target_net.load_state_dict(torch.load(self.FILEPATH + '/Model'))

        else:
            self.stepcount = 0
        

        #curriculum parameters
        self.gapsize = 0.002
        self.angle_range = 0.17
        self.speed_curriculum = [0.1, 0.01]


        if sensor == "fingertip":
            self.tactile_shape = (1,2,1,12)
        elif sensor == "plate":
            self.tactile_shape = (1,2,1,16,16)
        
        self.n_timesteps = n_timesteps
        self.cur_state_stack = State(state_myrmex=deque([torch.zeros(self.tactile_shape, dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_ep=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_jp=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_jt=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_jv=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps))


        #contains rewards & length of episode for every episode
     

        self.summary_writer = SummaryWriter(self.FILEPATH + '/runs')

        self.summary_writer.add_scalar('curriculum/max_gap', self.gapsize, self.stepcount)
        self.summary_writer.add_scalar('curriculum/angle_range', self.angle_range, self.stepcount)
    

    def obs_to_input(self, obs, cur_stack, device):

        if self.sensor == 'plate':
            cur_stack.state_myrmex.append(torch.cat([torch.from_numpy(obs['myrmex_r']).view(1, 1, 1,16,16),
                                                    torch.from_numpy(obs['myrmex_l']).view(1, 1, 1,16,16)], 
                                                    dim=1).to(device)) #.type(torch.DoubleTensor)
        elif self.sensor == 'fingertip':
            right = fingertip_hack(obs['myrmex_r'])
            left  = fingertip_hack(obs['myrmex_l'])
            cur_stack.state_myrmex.append(torch.cat([torch.from_numpy(right).view(1, 1, 1, 12),
                                                     torch.from_numpy(left).view(1, 1, 1, 12)], 
                                                     dim=1).to(device)) #.type(torch.DoubleTensor)
        
        cur_stack.state_ep.append(torch.from_numpy(obs["ee_pose"]).view(1, 1, 1, 7).to(device))                      #.type(torch.DoubleTensor)
        cur_stack.state_jp.append(torch.from_numpy(obs["joint_positions"]).view(1, 1, 1, 7).to(device))              #.type(torch.DoubleTensor)
        cur_stack.state_jt.append(torch.from_numpy(obs["joint_torques"]).view(1, 1, 1, 7).to(device))                #.type(torch.DoubleTensor)
        cur_stack.state_jv.append(torch.from_numpy(obs["joint_velocities"]).view(1, 1, 1, 7).to(device))             #.type(torch.DoubleTensor)
    
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
    
    def save_checkpoint(self, save_buffer=False):
        
        torch.save(self.policy_net.state_dict(), self.FILEPATH + '/Policy_Model')

        torch.save(self.target_net.state_dict(), self.FILEPATH +  '/Target_Model')       

        if save_buffer:
            with open(self.FILEPATH + 'exp_buffer.pickle', 'wb') as f:
                pickle.dump(self.replay_buffer, f)

        return 0
    
    def select_action(self, state, explore=True):
        
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.stepcount / self.EPS_DECAY)
        self.summary_writer.add_scalar('exploration/rate', eps_threshold, self.stepcount)
        
        if random.random() < eps_threshold and explore:
            action = torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
        else:
            action = self.policy_net(*state).max(1)[1].view(1,1)
        return action
    
    def _restart_env(self):
        
        self.env.close()
        self.env = gym.make('TactileObjectPlacementEnv-v0', continuous=False, sensor=self.sensor)

        return 0
    
    def _reset_env(self, options):

        success = False
        while not success:
            obs, info = self.env.reset(options=options)
            success = info['info']['success']

            if not success:
                self._restart_env()

        return obs, info

    def test(self, mode='max_gap'):
        
        if mode == 'max_gap':
            obs, info = self._reset_env(options={'gap_size' : self.gapsize, 'testing' : True, 'angle_range' : 0})

            if not info['success']:
                self._restart_env()

        elif mode == 'max_angle':
            g = self.gapsize - 0.002
            if g < 0.002:
                g = 0.002
            obs, info = self._reset_env(options={'gap_size' : g, 'testing' : True, 'angle_range' : self.angle_range})
        
        elif mode == 'random':
            obs, info = self._reset_env(options={'gap_size' : self.gapsize, 'testing' : False, 'angle_range' : self.angle_range})

        obs = self._normalize_observation(obs)

        done = False

        #ReInitialize cur_state_stack
        self.cur_state_stack = State(state_myrmex=deque([torch.zeros(self.tactile_shape, dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_ep=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_jp=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_jt=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_jv=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps))

        self.cur_state_stack = self.obs_to_input(obs["observation"], self.cur_state_stack, device=self.device)
        state = stack_to_state(self.cur_state_stack)

        for step in count():
            #experience sample: state, action, reward,  state+1
            print(step)
            action = self.select_action(state, explore=False)
            
            obs, reward, done, _ , info = self.env.step(action)
            obs = self._normalize_observation(obs)
            
            reward = torch.tensor([reward])
            if not done:
                self.cur_state_stack = self.obs_to_input(obs["observation"], self.cur_state_stack, device=self.device)
                next_state = stack_to_state(self.cur_state_stack)
            else:
                next_state = None

            state = next_state

            if done:
                break

        print('Finished ' , mode, ' Evaluation! Reward: ', float(reward), " Steps until Done: ", step+1)
        
        return reward, step+1

    def train(self):
        
        for episode in range(1, self.N_EPOCHS+1):
        
            obs, info = self._reset_env(options={'gap_size' : self.gapsize, 'testing' : False, 'angle_range' : self.angle_range})
            obs = self._normalize_observation(obs)

            done = False

            self.summary_writer.add_scalar('curriculum/sampled_gap', info['info']['sampled_gap'], episode)
            self.summary_writer.add_scalar('curriculum/sampled_angle', info['info']['obj_angle'], episode)
            
            #ReInitialize cur_state_stack
            self.cur_state_stack = State(state_myrmex=deque([torch.zeros(self.tactile_shape, dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_ep=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_jp=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_jt=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps),
                                     state_jv=deque([torch.zeros((1, 1, 1, 7), dtype=torch.double, device=self.device) for _ in range(self.n_timesteps)], maxlen=self.n_timesteps))

            self.cur_state_stack = self.obs_to_input(obs["observation"], self.cur_state_stack, device=self.device)
            state = stack_to_state(self.cur_state_stack)

            for step in count():
                #experience sample: state, action, reward,  state+1
                print(step)
                action = self.select_action(state)

                obs, reward, done, _ , info = self.env.step(action)
                obs = self._normalize_observation(obs)

                reward = torch.tensor([reward])
                if not done:
                    self.cur_state_stack = self.obs_to_input(obs["observation"], self.cur_state_stack, device=self.device)
                    next_state = stack_to_state(self.cur_state_stack)
                else:
                    next_state = None

                self.replay_buffer.push(state, action, next_state, reward)

                state = next_state

                self.optimize()
                self.stepcount += 1

                if done:
                    break
            
            print('Episode ', episode, ' done after ', step+1,  ' Steps ! reward: ', float(reward))
            
            self.summary_writer.add_scalar('Reward/train', reward, episode)
            self.summary_writer.add_scalar('Ep_length/train', step+1, episode)
            
            if episode % 100 == 0:
                
                R = []
                #3 Test episodes 
                for mode in ['max_gap', 'max_angle', 'random']:
                    r, s = self.test(mode)
                    R.append(r)
                    self.summary_writer.add_scalar('Reward/test/' + mode, r, episode)
                    self.summary_writer.add_scalar('Ep_length/test/' +mode , s, episode)
                
                #Gap is closed as soon reward is positive
                if R[0] >= 0.5 and R[-1] >= 0.5:   
                    self.gapsize += 0.002
                    if self.gapsize > 0.17:
                        self.gapsize = 0.17

                self.summary_writer.add_scalar('curriculum/max_gap', self.gapsize, episode)
                #only increase angle difficulty if the reward is high enough
                if R[1] >= 0.98 and R[-1] >= 0.98:
                    self.angle_range += 0.05
                    if self.angle_range > np.pi/2:
                        self.angle_range = np.pi/2
                    
                self.summary_writer.add_scalar('curriculum/angle_range', self.angle_range, episode)
                
                self.save_checkpoint()


        self.save_checkpoint(save_buffer=True)
        self.env.close()

    def optimize(self):

        transitions = self.replay_buffer.sample(self.BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        state_batch = State(*zip(*batch.state))

        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                    batch.next_state)), device=self.device, dtype=torch.bool)

        
        vals = self.policy_net(*(torch.cat(e) for e in state_batch))
        state_action_values = vals.gather(1, action_batch)
        
        
        #2. Compute Target Q-Values
        next_state_values = torch.zeros(action_batch.size()[0], device=self.device, dtype=torch.double)
        if any(non_final_mask):
            non_final_next_states = State(*zip(*[s for s in batch.next_state if s is not None]))
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(*(torch.cat(e) for e in non_final_next_states)).max(1)[0]

        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

        mean_Q_Vals = torch.mean(vals, dim=0)
        mean_sav = torch.mean(state_action_values, )
        self.summary_writer.add_scalar('Loss/train', loss, self.stepcount)

        for i, q in enumerate(mean_Q_Vals):
            label = 'Q_values/policy/action_' + str(i)
            self.summary_writer.add_scalar(label, q, self.stepcount)

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
    parser.add_argument('--mem_size', required=False, default='7500')
    parser.add_argument('--expl_slope', required=False, default='50000')
    parser.add_argument('--sensor', required=False, default='plate')

    opt = parser.parse_args()

    sensor = opt.sensor
    expl_slope = int(opt.expl_slope)
    mem_size = int(opt.mem_size)
    batchsize = int(opt.batchsize)
    nepochs = int(opt.nepochs)
    lr = float(opt.lr)

    #continue_ = int(opt.continue_training)

    time_string = datetime.now().strftime("%d-%m-%Y-%H:%M")

    if opt.savedir is None:
        filepath = '/homes/mjimenezhaertel/Masterarbeit/Training/' + time_string + '/'
    else:
        filepath = opt.savedir + time_string + '/'

    algo = DQN_Algo(filepath=filepath,
                    lr=lr, 
                    expl_slope=expl_slope, 
                    discount_factor=0.9, 
                    mem_size=mem_size, 
                    batch_size=batchsize, 
                    n_epochs=nepochs, 
                    tau=0.9,
                    n_timesteps=10,
                    sensor=sensor)

    algo.train()    
    algo.summary_writer.close()
