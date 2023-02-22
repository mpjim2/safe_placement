import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

#Basic Network, Only takes latest timestep as input
class place_net(nn.Module):
    
    def __init__(self, n_actions):
        super(place_net, self).__init__()
        
        #Convolutional network for feature extraction of Myrmex sensors
        self.myrmex_ftrs = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 8),
            torch.nn.Conv2d(8, 4, 4),
        )

        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(in_features=315, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_actions),
            torch.nn.Softmax(dim=0)
        )

    def forward(self, myrmex_r, 
                      myrmex_l, 
                      pose, 
                      j_pos, 
                      j_tau, 
                      j_vel):
        
        myr1 = torch.flatten(self.myrmex_ftrs(myrmex_r), start_dim=1)
        myr2 = torch.flatten(self.myrmex_ftrs(myrmex_l), start_dim=1)
        
        myr_c = torch.cat((myr1, myr2, pose, j_pos, j_tau, j_vel), 1)

        action_probs = self.feedforward(myr_c)

        
        return action_probs

#Takes history of last 20 Timesteps as input
class MAT_based_net(nn.Module):
    
    def __init__(self, n_actions, n_timesteps):
        super(MAT_based_net, self).__init__()

        self.myrmex_ftrs = torch.nn.Sequential(
            torch.nn.Conv2d(n_timesteps, 8, 8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 4, 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, 2),
            torch.nn.Tanh()
        )

        self.pose_embedding = torch.nn.Sequential(
            torch.nn.Linear(6*n_timesteps, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh()
        )
        self.jp_embedding = torch.nn.Sequential(
            torch.nn.Linear(7*n_timesteps, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh()
        )
        self.jt_embedding = torch.nn.Sequential(
            torch.nn.Linear(7*n_timesteps, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh()
        )
        self.jv_embedding = torch.nn.Sequential(
            torch.nn.Linear(7*n_timesteps, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh()
        )

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(100*2+128*4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_actions),
            torch.nn.ReLU()
        )
    
    def forward(self, myrmex_r, 
                      myrmex_l, 
                      pose, 
                      j_pos, 
                      j_tau, 
                      j_vel):


        ml_ftrs = torch.flatten(self.myrmex_ftrs(myrmex_l), start_dim=1)
        mr_ftrs = torch.flatten(self.myrmex_ftrs(myrmex_r), start_dim=1)

        combined_embedding = torch.cat([ml_ftrs, 
                                        mr_ftrs, 
                                        self.pose_embedding(pose),
                                        self.jp_embedding(j_pos),
                                        self.jt_embedding(j_tau),
                                        self.jv_embedding(j_vel)], dim=1)

        q_values = self.output_net(combined_embedding)
        
        return q_values
