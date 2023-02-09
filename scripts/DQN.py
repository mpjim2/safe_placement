import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class place_net(nn.Module):
    
    def __init__(self):
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
            torch.nn.Linear(64, 19),
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