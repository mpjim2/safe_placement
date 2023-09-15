import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class Conv2Plus1D(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_size, temporal_size, padding=0):
        """
          A sequence of convolutional layers that first apply the convolution operation over the
          spatial dimensions, and then the temporal dimension. 
        """
        super().__init__()
        self.seq = nn.Sequential(
            # Spatial decomposition
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=(1, spatial_size, spatial_size),
                      padding=padding),
            # Temporal decomposition
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, 
                      kernel_size=(temporal_size, 1, 1),
                      padding=padding)
        )
        
        #Size EE Pose: (batchsize, 1, 6, n_timesteps)

    def forward(self, x):
        return self.seq(x)

class dueling_placenet(nn.Module):
    
    def __init__(self, n_actions, n_timesteps, sensor_type, size, pitch_only, layersize) -> None:
        super().__init__()

        if sensor_type == 'plate':
            if size == 16:
                self.tactile_ftrs = nn.Sequential(
                    Conv2Plus1D(in_channels=2, 
                                out_channels=16,
                                spatial_size=7,
                                temporal_size=5),
                    nn.ReLU(),
                    Conv2Plus1D(in_channels=16,
                                out_channels=32,
                                spatial_size=5,
                                temporal_size=3),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(4608, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU()
                    )
            if size == 4:
                self.tactile_ftrs = nn.Sequential(
                    Conv2Plus1D(in_channels=2, 
                                out_channels=16,
                                spatial_size=2,
                                temporal_size=5),
                    nn.ReLU(),
                    Conv2Plus1D(in_channels=16,
                                out_channels=32,
                                spatial_size=2,
                                temporal_size=3),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(512, layersize),
                    nn.ReLU(),
                    nn.Linear(layersize, layersize),
                    nn.ReLU()
                )
        elif sensor_type =='fingertip':
            
            self.tactile_ftrs = nn.Sequential(
                nn.Conv2d(2, 16, (5, 1), padding=0, stride=1), 
                nn.ReLU(),
                nn.Conv2d(16, 32, (2, 1), padding=0, stride=1), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(640, layersize),
                nn.ReLU(),
                nn.Linear(layersize, layersize),
                nn.ReLU()
                )
        
        if not pitch_only:
            self.pose_embedding = nn.Sequential(
                nn.Conv2d(1, 16, (5, 1), padding=0, stride=1), 
                nn.ReLU(),
                nn.Conv2d(16, 32, (2, 1), padding=0, stride=1), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(640,layersize),
                nn.ReLU(),
                nn.Linear(layersize, layersize),
                nn.ReLU()
            )
        else:
            self.pose_embedding = nn.Sequential(
                nn.Conv2d(1, 8, (5, 1), padding=0, stride=1), 
                nn.ReLU(),
                nn.Conv2d(8, 16, (2, 1), padding=0, stride=1), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(80,layersize),
                nn.ReLU(),
                nn.Linear(layersize, layersize),
                nn.ReLU()
            )
        self.jt_embedding = nn.Sequential(
            nn.Conv2d(1, 16, (5, 1), padding=0, stride=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 1), padding=0, stride=1), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1120,layersize),
            nn.ReLU(),
            nn.Linear(layersize, layersize),
            nn.ReLU()
        )
        self.jv_embedding = nn.Sequential(
            nn.Conv2d(1, 16, (5, 1), padding=0, stride=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 1), padding=0, stride=1), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1120,layersize),
            nn.ReLU(),
            nn.Linear(layersize, layersize),
            nn.ReLU()
        )
        self.jp_embedding = nn.Sequential(
            nn.Conv2d(1, 16, (5, 1), padding=0, stride=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 1), padding=0, stride=1), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1120,layersize),
            nn.ReLU(),
            nn.Linear(layersize, layersize),
            nn.ReLU()
        )

        if sensor_type == 'fingertip':
            self.advantage_stream = torch.nn.Sequential(
                torch.nn.Linear(layersize*5, layersize),
                torch.nn.ReLU(),
                torch.nn.Linear(layersize, n_actions)
            )

            self.value_stream = torch.nn.Sequential(
                torch.nn.Linear(layersize*5, layersize),
                torch.nn.ReLU(),
                torch.nn.Linear(layersize, 1)
            )
        else:
            self.advantage_stream = torch.nn.Sequential(
                torch.nn.Linear(layersize*4+layersize, layersize),
                torch.nn.ReLU(),
                torch.nn.Linear(layersize, n_actions)
            )

            self.value_stream = torch.nn.Sequential(
                torch.nn.Linear(layersize*5, layersize),
                torch.nn.ReLU(),
                torch.nn.Linear(layersize, 1)
            )
    
    def forward(self, myrmex_data, 
                      pose, 
                      j_pos, 
                      j_tau, 
                      j_vel):
        
        combined_embedding = torch.cat([self.tactile_ftrs(myrmex_data), 
                                        self.pose_embedding(pose),
                                        self.jt_embedding(j_tau),
                                        self.jv_embedding(j_vel),
                                        self.jp_embedding(j_pos)
                                        ], dim=1)

        advantage = self.advantage_stream(combined_embedding)
        value     = self.value_stream(combined_embedding)
        
        q_values = value + (advantage - advantage.mean())
        
        return q_values

class dueling_placenet_TacTorqPose(nn.Module):
    
    def __init__(self, n_actions, n_timesteps, sensor_type, size, pitch_only) -> None:
        super().__init__()

        if sensor_type == 'plate':
            if size == 16:
                self.tactile_ftrs = nn.Sequential(
                    Conv2Plus1D(in_channels=2, 
                                out_channels=16,
                                spatial_size=7,
                                temporal_size=5),
                    nn.ReLU(),
                    Conv2Plus1D(in_channels=16,
                                out_channels=32,
                                spatial_size=5,
                                temporal_size=3),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(4608, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU()
                    )
            if size == 4:
                self.tactile_ftrs = nn.Sequential(
                    Conv2Plus1D(in_channels=2, 
                                out_channels=16,
                                spatial_size=2,
                                temporal_size=5),
                    nn.ReLU(),
                    Conv2Plus1D(in_channels=16,
                                out_channels=32,
                                spatial_size=2,
                                temporal_size=3),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU()
                )
        elif sensor_type =='fingertip':
            
            self.tactile_ftrs = nn.Sequential(
                nn.Conv2d(2, 16, (5, 1), padding=0, stride=1), 
                nn.ReLU(),
                nn.Conv2d(16, 32, (2, 1), padding=0, stride=1), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(640, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
                )
        
        if not pitch_only:
            self.pose_embedding = nn.Sequential(
                nn.Conv2d(1, 16, (5, 1), padding=0, stride=1), 
                nn.ReLU(),
                nn.Conv2d(16, 32, (2, 1), padding=0, stride=1), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(640,128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            )
        else:
            self.pose_embedding = nn.Sequential(
                nn.Conv2d(1, 8, (5, 1), padding=0, stride=1), 
                nn.ReLU(),
                nn.Conv2d(8, 16, (2, 1), padding=0, stride=1), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(80,128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            )
        self.jt_embedding = nn.Sequential(
            nn.Conv2d(1, 16, (5, 1), padding=0, stride=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 1), padding=0, stride=1), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1120,128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        if sensor_type == 'fingertip':
            self.advantage_stream = torch.nn.Sequential(
                torch.nn.Linear(128*3, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, n_actions)
            )

            self.value_stream = torch.nn.Sequential(
                torch.nn.Linear(128*3, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1)
            )
        else:
            self.advantage_stream = torch.nn.Sequential(
                torch.nn.Linear(128*2+128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, n_actions)
            )

            self.value_stream = torch.nn.Sequential(
                torch.nn.Linear(128*2+128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1)
            )
    
    def forward(self, myrmex_data, 
                      pose, 
                      j_pos, 
                      j_tau, 
                      j_vel):
        
        combined_embedding = torch.cat([self.tactile_ftrs(myrmex_data), 
                                        self.pose_embedding(pose),
                                        self.jt_embedding(j_tau)
                                        ], dim=1)

        advantage = self.advantage_stream(combined_embedding)
        value     = self.value_stream(combined_embedding)
        
        q_values = value + (advantage - advantage.mean())
        
        return q_values

class dueling_placenet_TacTorqRot(nn.Module):
    
    def __init__(self, n_actions, n_timesteps, sensor_type, size, pitch_only, layersize) -> None:
        super().__init__()

        if sensor_type == 'plate':
            if size == 16:
                self.tactile_ftrs = nn.Sequential(
                    Conv2Plus1D(in_channels=2, 
                                out_channels=16,
                                spatial_size=7,
                                temporal_size=5),
                    nn.ReLU(),
                    Conv2Plus1D(in_channels=16,
                                out_channels=32,
                                spatial_size=5,
                                temporal_size=3),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(4608, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU()
                    )
            if size == 4:
                self.tactile_ftrs = nn.Sequential(
                    Conv2Plus1D(in_channels=2, 
                                out_channels=16,
                                spatial_size=2,
                                temporal_size=5),
                    nn.ReLU(),
                    Conv2Plus1D(in_channels=16,
                                out_channels=32,
                                spatial_size=2,
                                temporal_size=3),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU()
                )
        elif sensor_type =='fingertip':
            test = nn.Sequential(
                    nn.Conv2d(2, 16, (5, 1), padding=0), 
                    nn.ReLU(),
                    nn.Conv2d(16, 32, (2, 1), padding=0), 
                    nn.ReLU(),
                    nn.Flatten())
            
            x  = torch.rand((1, 2, n_timesteps, 4))

            num_features = test(x).size()[1]

            self.tactile_ftrs = nn.Sequential(
                nn.Conv2d(2, 16, (5, 1), padding=0, stride=1), 
                nn.ReLU(),
                nn.Conv2d(16, 32, (2, 1), padding=0, stride=1), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(num_features, layersize),
                nn.ReLU(),
                nn.Linear(layersize, layersize),
                nn.ReLU()
                )
        
        if not pitch_only:
            self.pose_embedding = nn.Sequential(
                nn.Conv2d(1, 16, (5, 1), padding=0, stride=1), 
                nn.ReLU(),
                nn.Conv2d(16, 32, (2, 1), padding=0, stride=1), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(640,layersize),
                nn.ReLU(),
                nn.Linear(layersize, layersize),
                nn.ReLU()
            )
        else:
            self.pose_embedding = nn.Sequential(
                nn.Conv2d(1, 8, (5, 1), padding=0, stride=1), 
                nn.ReLU(),
                nn.Conv2d(8, 16, (2, 1), padding=0, stride=1), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(80,layersize),
                nn.ReLU(),
                nn.Linear(layersize, layersize),
                nn.ReLU()
            )

        self.jt_embedding = nn.Sequential(
            nn.Conv2d(1, 16, (5, 1), padding=0, stride=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 1), padding=0, stride=1), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1120,layersize),
            nn.ReLU(),
            nn.Linear(layersize, layersize),
            nn.ReLU()
        )

        if sensor_type == 'fingertip':

            self.advantage_stream = torch.nn.Sequential(
                torch.nn.Linear(layersize*3, layersize),
                torch.nn.ReLU(),
                torch.nn.Linear(layersize, n_actions)
            )

            self.value_stream = torch.nn.Sequential(
                torch.nn.Linear(layersize*3, layersize),
                torch.nn.ReLU(),
                torch.nn.Linear(layersize, 1)
            )
        else:
            self.advantage_stream = torch.nn.Sequential(
                torch.nn.Linear(128+64+16, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, n_actions)
            )

            self.value_stream = torch.nn.Sequential(
                torch.nn.Linear(layersize*3, layersize),
                torch.nn.ReLU(),
                torch.nn.Linear(layersize, 1)
            )
    def forward(self, myrmex_data, 
                      pose, 
                      j_pos, 
                      j_tau, 
                      j_vel):
        
        combined_embedding = torch.cat([self.tactile_ftrs(myrmex_data), 
                                        self.pose_embedding(pose),
                                        self.jt_embedding(j_tau)
                                        ], dim=1)

        advantage = self.advantage_stream(combined_embedding)
        value     = self.value_stream(combined_embedding)
        
        q_values = value + (advantage - advantage.mean())
        
        return q_values

class dueling_placenet_TacRot(nn.Module):
    
    def __init__(self, n_actions, n_timesteps, sensor_type, size, pitch_only) -> None:
        super().__init__()

        if sensor_type == 'plate':
            if size == 16:
                self.tactile_ftrs = nn.Sequential(
                    Conv2Plus1D(in_channels=2, 
                                out_channels=16,
                                spatial_size=7,
                                temporal_size=5),
                    nn.ReLU(),
                    Conv2Plus1D(in_channels=16,
                                out_channels=32,
                                spatial_size=5,
                                temporal_size=3),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(4608, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU()
                    )
            if size == 4:
                self.tactile_ftrs = nn.Sequential(
                    Conv2Plus1D(in_channels=2, 
                                out_channels=16,
                                spatial_size=2,
                                temporal_size=5),
                    nn.ReLU(),
                    Conv2Plus1D(in_channels=16,
                                out_channels=32,
                                spatial_size=2,
                                temporal_size=3),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU()
                )
        elif sensor_type =='fingertip':
            test = nn.Sequential(
                    nn.Conv2d(2, 16, (5, 1), padding=0), 
                    nn.ReLU(),
                    nn.Conv2d(16, 32, (2, 1), padding=0), 
                    nn.ReLU(),
                    nn.Flatten())
            
            x  = torch.rand((1, 2, n_timesteps, 4))
            num_features = test(x).size()[1]

            print(num_features)
            self.tactile_ftrs = nn.Sequential(
                nn.Conv2d(2, 16, (5, 1), padding=0, stride=1), 
                nn.ReLU(),
                nn.Conv2d(16, 32, (2, 1), padding=0, stride=1), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(num_features, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
                )
        if pitch_only:
            self.pose_embedding = nn.Sequential(
                nn.Conv2d(1, 16, (5, 1), padding=0, stride=1), 
                nn.ReLU(),
                nn.Conv2d(16, 32, (2, 1), padding=0, stride=1), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(160,128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            )
        else:
            self.pose_embedding = nn.Sequential(
                nn.Conv2d(1, 16, (5, 1), padding=0, stride=1), 
                nn.ReLU(),
                nn.Conv2d(16, 32, (2, 1), padding=0, stride=1), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(640,128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            )
        self.advantage_stream = torch.nn.Sequential(
            torch.nn.Linear(128*2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_actions)
        )

        self.value_stream = torch.nn.Sequential(
            torch.nn.Linear(128*2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
     
    def forward(self, myrmex_data, 
                      pose, 
                      j_pos, 
                      j_tau, 
                      j_vel):
        
        combined_embedding = torch.cat([self.tactile_ftrs(myrmex_data), 
                                        self.pose_embedding(pose)
                                        ], dim=1)

        advantage = self.advantage_stream(combined_embedding)
        value     = self.value_stream(combined_embedding)
        
        q_values = value + (advantage - advantage.mean())
        
        return q_values
    
class dueling_placenet_tactileonly(nn.Module):
    
    def __init__(self, n_actions, n_timesteps, sensor_type, size, layersize) -> None:
        super().__init__()

        if sensor_type == 'plate':
            if size == 16:
                self.tactile_ftrs = nn.Sequential(
                    Conv2Plus1D(in_channels=2, 
                                out_channels=16,
                                spatial_size=7,
                                temporal_size=5),
                    nn.ReLU(),
                    Conv2Plus1D(in_channels=16,
                                out_channels=32,
                                spatial_size=5,
                                temporal_size=3),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(4608, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU()
                    )
            if size == 4:
                self.tactile_ftrs = nn.Sequential(
                    Conv2Plus1D(in_channels=2, 
                                out_channels=16,
                                spatial_size=2,
                                temporal_size=5),
                    nn.ReLU(),
                    Conv2Plus1D(in_channels=16,
                                out_channels=32,
                                spatial_size=2,
                                temporal_size=3),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU()
                )
        elif sensor_type =='fingertip':
            
            test = nn.Sequential(
                    nn.Conv2d(2, 16, (5, 1), padding=0), 
                    nn.ReLU(),
                    nn.Conv2d(16, 32, (2, 1), padding=0), 
                    nn.ReLU(),
                    nn.Flatten())
            
            x  = torch.rand((1, 2, n_timesteps, 4))

            num_features = test(x).size()[1]
            print(num_features)
            self.tactile_ftrs = nn.Sequential(
                nn.Conv2d(2, 16, (5, 1), padding=0), 
                nn.ReLU(),
                nn.Conv2d(16, 32, (2, 1), padding=0), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(num_features, layersize),
                nn.ReLU(),
                nn.Linear(layersize, layersize),
                nn.ReLU()
                )
        

        if sensor_type == 'fingertip':
            self.advantage_stream = torch.nn.Sequential(
                torch.nn.Linear(layersize, layersize),
                torch.nn.ReLU(),
                torch.nn.Linear(layersize, n_actions)
            )

            self.value_stream = torch.nn.Sequential(
                torch.nn.Linear(layersize, layersize),
                torch.nn.ReLU(),
                torch.nn.Linear(layersize, 1)
            )
        else:
            self.advantage_stream = torch.nn.Sequential(
                torch.nn.Linear(layersize, layersize),
                torch.nn.ReLU(),
                torch.nn.Linear(layersize, n_actions)
            )

            self.value_stream = torch.nn.Sequential(
                torch.nn.Linear(layersize, layersize),
                torch.nn.ReLU(),
                torch.nn.Linear(layersize, 1)
            )
    def forward(self, myrmex_data, 
                      pose, 
                      j_pos, 
                      j_tau, 
                      j_vel):
        
        tactile_embedding = self.tactile_ftrs(myrmex_data) 
                            
        advantage = self.advantage_stream(tactile_embedding)
        value     = self.value_stream(tactile_embedding)
        
        q_values = value + (advantage - advantage.mean())
        
        return q_values


if __name__=='__main__':
 
   
    fullS = dueling_placenet(n_actions=5, n_timesteps=10, sensor_type='fingertip', size=4, pitch_only=True, layersize=16)

    fullM = dueling_placenet(n_actions=5, n_timesteps=10, sensor_type='fingertip', size=4, pitch_only=True, layersize=32)

    tacS = dueling_placenet_tactileonly(n_actions=5, n_timesteps=10, sensor_type='fingertip', size=4, layersize=16)

    tacM = dueling_placenet_tactileonly(n_actions=5, n_timesteps=10, sensor_type='fingertip', size=4, layersize=32)

    ttrS = dueling_placenet_TacTorqRot(n_actions=5, n_timesteps=10, sensor_type='fingertip', size=4, pitch_only=True, layersize=16)

    ttrM = dueling_placenet_TacTorqRot(n_actions=5, n_timesteps=10, sensor_type='fingertip', size=4, pitch_only=True, layersize=32)

