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


def NormalizeData(data, high, low):

    return (2 * ((data - low) / (high - low))) -1

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

#### DATA COMES FROM ENV AS : (n_time_steps, datadim) for myrmex : (10, 4) x 2 --turnto-->  (1, 2, 10, 4)
##  FOR ANY OTHER (10,7) ---> (1, 1, 10, 7) is required

class DQN_Algo():

    def __init__(self, filepath, lr=1e5, expl_slope=1000, discount_factor=0.95, mem_size=10000, batch_size=128, n_epochs=1000, tau=0.9, n_timesteps=10, episode=0, sensor="plate", global_step=None, architecture='temp_conv', reduced=0, grid_size=16, actionspace='full', eval=False):

        self.sensor = sensor
        self.FILEPATH = filepath 
        if not eval:
            time = filepath.split('/')[-2]
            tb_runname = 'sensor' + sensor + '_Arch' + architecture + '_reduced' +str(reduced) + '_' + time
        
        
        self.env = gym.make('TactileObjectPlacementEnv-v1', continuous=True, sensor=sensor, grid_size=grid_size, action_space=actionspace, timesteps=n_timesteps)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.START_EP = episode
        #Policy and target network initilisation
        if architecture == 'temp_conv':
            if reduced == 1:
                self.policy_net = DQN.dueling_placenet_TacTorqRot(n_actions=self.env.action_space.n, n_timesteps=n_timesteps, sensor_type=sensor, size=grid_size).double().to(self.device)
                self.target_net = DQN.dueling_placenet_TacTorqRot(n_actions=self.env.action_space.n, n_timesteps=n_timesteps, sensor_type=sensor, size=grid_size).double().to(self.device)
                self.target_net.load_state_dict(self.policy_net.state_dict())
            elif reduced == 0:
                self.policy_net = DQN.dueling_placenet(n_actions=self.env.action_space.n, n_timesteps=n_timesteps, sensor_type=sensor, size=grid_size).double().to(self.device)
                self.target_net = DQN.dueling_placenet(n_actions=self.env.action_space.n, n_timesteps=n_timesteps, sensor_type=sensor, size=grid_size).double().to(self.device)
                self.target_net.load_state_dict(self.policy_net.state_dict())
            elif reduced == 2: 
                self.policy_net = DQN.dueling_placenet_TacRot(n_actions=self.env.action_space.n, n_timesteps=n_timesteps, sensor_type=sensor, size=grid_size).double().to(self.device)
                self.target_net = DQN.dueling_placenet_TacRot(n_actions=self.env.action_space.n, n_timesteps=n_timesteps, sensor_type=sensor, size=grid_size).double().to(self.device)
                self.target_net.load_state_dict(self.policy_net.state_dict())
            elif reduced == 3:
                self.policy_net = DQN.dueling_placenet_tactileonly(n_actions=self.env.action_space.n, n_timesteps=n_timesteps, sensor_type=sensor, size=grid_size).double().to(self.device)
                self.target_net = DQN.dueling_placenet_tactileonly(n_actions=self.env.action_space.n, n_timesteps=n_timesteps, sensor_type=sensor, size=grid_size).double().to(self.device)
                self.target_net.load_state_dict(self.policy_net.state_dict())
            elif reduced == 4:
                self.policy_net = DQN.dueling_placenet_TacTorque(n_actions=self.env.action_space.n, n_timesteps=n_timesteps, sensor_type=sensor, size=grid_size).double().to(self.device)
                self.target_net = DQN.dueling_placenet_TacTorque(n_actions=self.env.action_space.n, n_timesteps=n_timesteps, sensor_type=sensor, size=grid_size).double().to(self.device)
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
        
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = expl_slope
        
        self.discount_factor  = discount_factor
        self.BATCH_SIZE = batch_size
        self.N_EPOCHS = n_epochs

        self.LEARNING_RATE = lr

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.LEARNING_RATE, amsgrad=False)
        # self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.LEARNING_RATE, amsgrad=True)

        self.soft_update_weight = tau
        
        self.max_ep_steps = 31

        if not global_step is None or eval:
            self.stepcount=global_step if global_step is not None else 0 
            #load NN Dictionary
            self.policy_net.load_state_dict(torch.load(self.FILEPATH + '/Policy_Model'))
            self.target_net.load_state_dict(torch.load(self.FILEPATH + '/Target_Model'))
            if not eval:
                with open(self.FILEPATH + '/exp_buffer.pickle', 'rb') as f:
                    self.replay_buffer = pickle.load(f)
        else:
            self.stepcount = 0
            
            self.replay_buffer = ReplayMemory(mem_size)

        self.grid_size = grid_size
        #curriculum parameters
        
        self.I = 0
        self.step_vals = [11, 23, 33, 38]

        self.gapsize = 0.002

        if actionspace == 'reduced':
            self.angle_range = 0.0
        elif actionspace =='full':
            self.angle_range = 0.175

        self.sim_steps = 50

        self.train_avg = 0
        self.test_avg = 0

        self.reduced = reduced
        
        if actionspace == 'full':

            self.reward_fn = 'close_gap'
        else:
            self.reward_fn = 'place'
            
        if sensor == "fingertip":
            self.tactile_shape = (1,2,1,4)
        elif sensor == "plate":
            self.tactile_shape = (1,2,1,grid_size,grid_size)
        
        self.n_timesteps = n_timesteps
        #contains rewards & length of episode for every episode
     

        if not eval:
            self.summary_writer = SummaryWriter('/home/marco/Masterarbeit/Training/AllRuns/TB_logs/Experiments_FinallyWorking/' + tb_runname)

            self.summary_writer.add_scalar('curriculum/max_gap', self.gapsize, self.stepcount)
            self.summary_writer.add_scalar('curriculum/angle_range', self.angle_range, self.stepcount)
    
        self.tactile_test = None

    def obs_to_input(self, obs, device):

        if self.sensor == 'plate':
            myrmex_r = obs['myrmex_r']
            myrmex_l = obs['myrmex_l']

            combined = torch.cat([torch.from_numpy(myrmex_r).view(1, 1, self.n_timesteps, 16, 16),
                                  torch.from_numpy(myrmex_l).view(1, 1, self.n_timesteps, 16, 16)], 
                                  dim=1).to(device)

            
        elif self.sensor == 'fingertip':
            #incoming 
            myrmex_r = obs['myrmex_r']
            myrmex_l = obs['myrmex_l']

            combined = torch.cat([torch.from_numpy(myrmex_r).view(1, 1, self.n_timesteps, 4),
                                  torch.from_numpy(myrmex_l).view(1, 1, self.n_timesteps, 4)], 
                                  dim=1).to(device)

        pose = torch.from_numpy(obs['ee_pose']).view(1, 1, self.n_timesteps, 4)  

        jp = torch.from_numpy(obs['joint_positions']).view(1, 1, self.n_timesteps, 7)
        jt = torch.from_numpy(obs['joint_torques']).view(1, 1, self.n_timesteps, 7)
        jv = torch.from_numpy(obs['joint_velocities']).view(1, 1, self.n_timesteps, 7)
        state = State(state_myrmex=combined, state_ep=pose, state_jp=jp, state_jt=jt, state_jv=jv)

        return state 

    def _normalize_observation(self, obs):
        
        normalized = {'observation' : {}}
        for key in obs['observation']:
            if not(key == "myrmex_r" or key == 'myrmex_l'):
                min_ = self.env.spaces[key].high
                max_ = self.env.spaces[key].low
                
                if key == 'ee_pose':
                    min_ = min_[:3]
                    max_ = max_[:3]

                    # pos  = NormalizeData(obs['observation'][key][:, :3], min_, max_)
                    quat = obs['observation'][key][:, 3:] 

                    normalized['observation'][key] = quat
                else:
                    normalized['observation'][key] = NormalizeData(obs['observation'][key], min_, max_)
        
        normalized['observation']['myrmex_r'] = obs['observation']['myrmex_r']
        normalized['observation']['myrmex_l'] = obs['observation']['myrmex_l']

        return normalized
    
    def save_checkpoint(self, save_buffer=False):
        
        torch.save(self.policy_net.state_dict(), self.FILEPATH + 'Policy_Model')

        torch.save(self.target_net.state_dict(), self.FILEPATH +  'Target_Model')       

        if save_buffer:
            with open(self.FILEPATH + 'exp_buffer.pickle', 'wb') as f:
                pickle.dump(self.replay_buffer, f)

        return 0
    
    def select_action(self, state, explore=True):
        
        #if explore:
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.stepcount / self.EPS_DECAY)
            #self.summary_writer.add_scalar('exploration/rate', eps_threshold, self.stepcount)
        
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

    def test(self, log_actions=False, test_action=None, evaluate=False):
        
        obs, info = self._reset_env(options={'gap_size' : self.gapsize, 'testing' : True, 'angle_range' : self.angle_range, 'sim_steps' : self.sim_steps, 'reward_fn' :self.reward_fn})

        obs = self._normalize_observation(obs)
        
        state = self.obs_to_input(obs["observation"], device=self.device)
        done = False

        if evaluate:
            qvals = []
            RIGHT = []
            LEFT  = []
        cumulative_reward = 0
        for step in count():
            #experience sample: state, action, reward,  state+1
            if evaluate:
                q = self.policy_net(*state).detach().numpy()
                qvals.append(q[0])
            if test_action is None:  
                action = self.select_action(state, explore=False)
            else:
                action = torch.tensor([[test_action]], device=self.device, dtype=torch.long)

            print(self.env.DISCRETE_ACTIONS[action])
            if log_actions:
                a_vec = self.env.DISCRETE_ACTIONS[action]

                self.summary_writer.add_scalar("test/actions/Z_translation", a_vec[0], self.stepcount+step)
                self.summary_writer.add_scalar("test/actions/X_rotation", a_vec[1], self.stepcount+step)


            obs, reward, done, _ , info = self.env.step(action)
            obs = self._normalize_observation(obs)
            
            if evaluate:
                L = obs['observation']['myrmex_l']
                R = obs['observation']['myrmex_r']

                for l in L:
                    LEFT.append(l)

                for r in R:
                    RIGHT.append(r)

            cumulative_reward += reward
            reward = torch.tensor([reward])
            if not done and step < self.max_ep_steps:
                next_state = self.obs_to_input(obs["observation"], device=self.device)
            else:
                next_state = None

            state = next_state
            if done:
                break
            if step >= self.max_ep_steps:
                reward = -1
                break
            
        print('Finished Evaluation! Reward: ', float(reward), " Steps until Done: ", step+1, ' Termination Cause: ' , info['cause'])
        
        if evaluate:
            return qvals, (LEFT, RIGHT)
        else:
            return reward, cumulative_reward, step+1

    def train(self):
        
        for episode in range(self.START_EP+1, self.START_EP + self.N_EPOCHS+1):
            
            obs, info = self._reset_env(options={'gap_size' : self.gapsize, 'testing' : False, 'angle_range' : self.angle_range, 'sim_steps' : self.sim_steps, 'reward_fn' : self.reward_fn})
            obs = self._normalize_observation(obs)

            done = False

            self.summary_writer.add_scalar('curriculum/sampled_gap', info['info']['sampled_gap'], episode)
            self.summary_writer.add_scalar('curriculum/sampled_angle', info['info']['obj_angle'], episode)
            
            #ReInitialize cur_state_stack

            state = self.obs_to_input(obs["observation"], device=self.device)

            if episode == 1:
                self.summary_writer.add_graph(self.policy_net, state)

            pre_buffer = []
            cumulative_reward = 0
            for step in count():

                #experience sample: state, action, reward,  state+1
                if step >= self.max_ep_steps:
                    action = torch.tensor([[self.env.action_space.n - 1]], device=self.device, dtype=torch.long)
                else:
                    action = self.select_action(state)
                         
                obs, reward, done, _ , info = self.env.step(action)
                obs = self._normalize_observation(obs)

                cumulative_reward += reward
                
                if not done and step < self.max_ep_steps:
                    next_state = self.obs_to_input(obs["observation"], device=self.device)
                else:
                    next_state = None
                
                reward = torch.tensor([reward])
                pre_buffer.append((state, action, next_state, reward))
                #self.replay_buffer.push(state, action, next_state, reward)

                state = next_state

               
                if done:
                    if reward > 0:
                        if step >= (self.step_vals[self.I] / (self.sim_steps/10)):
                            self.summary_writer.add_scalar('Reward/train/final', reward, episode)
                            self.summary_writer.add_scalar('Ep_length/train', step+1, episode)
                            self.summary_writer.add_scalar('Reward/train/Cumulative', cumulative_reward, episode)
                            for transition in pre_buffer:
                                self.replay_buffer.push(*transition)
                                self.optimize()
                                self.stepcount += 1    
                    else:
                        self.summary_writer.add_scalar('Reward/train/final', reward, episode)
                        self.summary_writer.add_scalar('Ep_length/train', step+1, episode)    
                        self.summary_writer.add_scalar('Reward/train/Cumulative', cumulative_reward, episode)  
                        for transition in pre_buffer:
                            self.replay_buffer.push(*transition)
                            self.optimize()
                            self.stepcount += 1
                    break
            
            print('Episode ', episode, ' done after ', step+1,  ' Steps ! reward: ', float(reward))

            if episode % 100 == 0:
                
                R = []
                S = []
                C = []
                #3 Test episodes 
   
                for i in range(3):
                    r, c, s = self.test(log_actions = i==0)
                    R.append(float(r))
                    S.append(int(s))
                    C.append(float(c))
                    self.summary_writer.add_scalar('Reward/test/final', r, episode+i)
                    self.summary_writer.add_scalar('Ep_length/test/', s, episode+i)
                    
                mr = np.mean(R)
                ms = np.mean(S)
                mc = np.mean(C)

                self.summary_writer.add_scalar('Reward/test/mean', mr, episode)
                self.summary_writer.add_scalar('Reward/test/meanCumulative/', mc, episode)
                
                #Curriculum: first increase gapsize then increase angle range
                if self.sim_steps > 10:
                    if mr >= 0:   
                        self.sim_steps -= 10
                        if self.sim_steps < 10:
                            self.sim_steps = 10
            
                elif self.gapsize < 0.015:
                    if mr >= 0:   
                        self.gapsize += 0.005
                        if self.I < len(self.step_vals)-1:
                            self.I += 1
                        if self.gapsize > 0.015:
                            self.gapsize = 0.015
                            self.reward_fn = 'place'
                        if self.max_ep_steps < 150:
                            self.max_ep_steps = self.step_vals[self.I] + 20
                # else:
                #     if mr >= 0.9:
                #         self.angle_range += 0.05
                #         if self.angle_range > np.pi/2:
                #             self.angle_range = np.pi/2
                #         # if self.max_ep_steps < 1000:
                #         #     self.max_ep_steps += 50
                
                self.summary_writer.add_scalar('curriculum/max_gap', self.gapsize, episode)
                #only increase angle difficulty if the reward is high enough
                    
                self.summary_writer.add_scalar('curriculum/angle_range', self.angle_range, episode)
                
                self.save_checkpoint(save_buffer=True)


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
        #Double DQN: Decouble action selection and value estimation
        #Use policy network to choose best action
        #use target network to estimate value of that action 
        next_state_values = torch.zeros(action_batch.size()[0], device=self.device, dtype=torch.double)
        if any(non_final_mask):
            
            non_final_next_states = State(*zip(*[s for s in batch.next_state if s is not None]))
            
            with torch.no_grad():
                #simple max of q vals. Target net is used for selection and evaluation
                # next_state_values[non_final_mask] = self.target_net(*(torch.cat(e) for e in non_final_next_states)).max(1)[0]
                
                #double DQN: Select action using policy net; evaluate using target net
                next_state_values_ = self.target_net(*(torch.cat(e) for e in non_final_next_states))        
                next_state_vs = self.policy_net(*(torch.cat(e) for e in non_final_next_states))
                next_state_actions = torch.argmax(next_state_vs, dim=1).unsqueeze(dim=1)

                next_state_values[non_final_mask] = next_state_values_.gather(1, next_state_actions).squeeze()

        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

        mean_Q_Vals = torch.mean(vals, dim=0)
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
                    n_timesteps=10,
                    sensor=sensor,
                    global_step=global_step,
                    architecture=opt.architecture,
                    reduced=reduced,
                    episode = episode,
                    actionspace=actionspace)

    algo.train()    
    algo.summary_writer.close()
