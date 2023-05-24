import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

#Basic Network, Only takes latest timestep as input

#Takes history of last n Timesteps as input
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
            torch.nn.Linear(7*n_timesteps, 128),
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
            torch.nn.Linear(128, n_actions)   
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


# class ResidualBlock(nn.Module):
    
#     def __init__(self, in_channels, out_channels, spatial_size, temporal_size) -> None:
#         super().__init__()
#         self.seq = nn.Sequential(
#             Conv2Plus1D(in_channels, out_channels, spatial_size, temporal_size)


#         )
        
#     def forward(x):

#         return self.seq(x)

class LSTM_custom(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):

        final_hidden, (_,_) = self.lstm(x)

        return final_hidden[:,-1,:]
        

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

class placenet_v2(nn.Module):

    def __init__(self, n_actions, n_timesteps, sensor_type, size) -> None:
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
                nn.Conv2d(2, 1, (5, 1), padding=0), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(72, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
                )
            
        self.pose_embedding = nn.Sequential(
            nn.Conv2d(1, 1, (5, 1), padding=0), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(42,128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.jp_embedding = nn.Sequential(
            nn.Conv2d(1, 1, (5, 1), padding=0), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(42,128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        self.jv_embedding = nn.Sequential(
            nn.Conv2d(1, 1, (5, 1), padding=0), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(42,128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        self.jt_embedding = nn.Sequential(
            nn.Conv2d(1, 1, (5, 1), padding=0), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(42,128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(128*5, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_actions)
        )

    def forward(self, myrmex_data, 
                      pose, 
                      j_pos, 
                      j_tau, 
                      j_vel):
        
        combined_embedding = torch.cat([self.tactile_ftrs(myrmex_data), 
                                        self.pose_embedding(pose),
                                        self.jp_embedding(j_pos),
                                        self.jt_embedding(j_tau),
                                        self.jv_embedding(j_vel)], dim=1)

        q_values = self.output_net(combined_embedding)
        
        return q_values


class placenet_LSTM(nn.Module):

    def __init__(self, n_actions, n_timesteps, sensor_type) -> None:
        super().__init__()

        if sensor_type == 'plate':
            self.tactile_ftrs = nn.Sequential(
                Conv2Plus1D(in_channels=2, 
                            out_channels=16,
                            spatial_size=7,
                            temporal_size=1),
                nn.ReLU(),
                Conv2Plus1D(in_channels=16,
                            out_channels=32,
                            spatial_size=5,
                            temporal_size=1),
                nn.ReLU(),
                Conv2Plus1D(in_channels=32,
                            out_channels=1,
                            spatial_size=3,
                            temporal_size=1),
                nn.ReLU(),
                nn.Flatten(start_dim=-2, end_dim=-1),
                nn.Flatten(start_dim=1, end_dim=2),
                LSTM_custom(16, 128), 
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
                )
        elif sensor_type =='fingertip':
            
            self.tactile_ftrs = nn.Sequential(
                nn.Conv2d(2, 1, (1, 1), padding=0), 
                nn.ReLU(),
                nn.Flatten(start_dim=1, end_dim=2),
                LSTM_custom(12,128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
                )
            
        self.pose_embedding = nn.Sequential(
                nn.Flatten(start_dim=1, end_dim=2),
                LSTM_custom(7,128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
                )

        self.jp_embedding = nn.Sequential(
                nn.Flatten(start_dim=1, end_dim=2),
                LSTM_custom(7,128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
                )
        
        self.jv_embedding = nn.Sequential(
                nn.Flatten(start_dim=1, end_dim=2),
                LSTM_custom(7,128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
                )
        
        self.jt_embedding = nn.Sequential(
                nn.Flatten(start_dim=1, end_dim=2),
                LSTM_custom(7,128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
                )

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(128*5, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_actions)
        )

    def forward(self, myrmex_data, 
                      pose, 
                      j_pos, 
                      j_tau, 
                      j_vel):
 

        combined_embedding = torch.cat([self.tactile_ftrs(myrmex_data), 
                                        self.pose_embedding(pose),
                                        self.jp_embedding(j_pos),
                                        self.jt_embedding(j_tau),
                                        self.jv_embedding(j_vel)], dim=1)

        q_values = self.output_net(combined_embedding)
        
        return q_values



class dueling_placenet(nn.Module):
    
    def __init__(self, 
                 n_actions, 
                 n_timesteps, 
                 sensor_type,
                 pose=True,
                 torques=True,
                 velocities=True,
                 positions=True,
                 layersize=128
                ) -> None:
        super().__init__()


        self.pose = pose
        self.torques = torques
        self.velocities = velocities
        self.positions = positions

        n_heads = 1
        if sensor_type == 'plate':
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
                nn.Linear(4608, layersize),
                nn.ReLU(),
                nn.Linear(layersize, layersize),
                nn.ReLU()
                )
           
        elif sensor_type =='fingertip':
            
            size_calculator = nn.Sequential(
                                nn.Conv2d(2, 4, (5, 1), padding=0), 
                                nn.ReLU(),
                                nn.Conv2d(4, 8, (2, 1), padding=0), 
                                nn.ReLU(),
                                nn.Flatten()
                                )
            x_1  = torch.rand((1, 1, n_timesteps, 4))
            x_2  = torch.rand((1, 1, n_timesteps, 7))
            tactile_size = size_calculator(x_1).size()[1]
            joint_state_size = size_calculator(x_2).size()[1]
            
            
            self.tactile_ftrs = nn.Sequential(
                nn.Conv2d(2, 4, (5, 1), padding=0), 
                nn.ReLU(),
                nn.Conv2d(4, 8, (2, 1), padding=0), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(28, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
                )
        
        if pose:
            
            self.pose_embedding = nn.Sequential(
                nn.Conv2d(1, 1, (3, 1), padding=0), 
                nn.ReLU(),
                nn.Conv2d(1, 1, (2, 1), padding=0), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(tactile_size,layersize),
                nn.ReLU(),
                nn.Linear(layersize, layersize),
                nn.ReLU()
            )
            n_heads += 1
        
        if positions:
            self.jp_embedding = nn.Sequential(
                nn.Conv2d(1, 1, (3, 1), padding=0), 
                nn.ReLU(),
                nn.Conv2d(1, 1, (2, 1), padding=0), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(joint_state_size,layersize),
                nn.ReLU(),
                nn.Linear(layersize, layersize),
                nn.ReLU()
            )
            n_heads += 1
        
        if velocities:
            self.jv_embedding = nn.Sequential(
                nn.Conv2d(1, 1, (3, 1), padding=0), 
                nn.ReLU(),
                nn.Conv2d(1, 1, (2, 1), padding=0), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(joint_state_size,layersize),
                nn.ReLU(),
                nn.Linear(layersize, layersize),
                nn.ReLU()
            )
            n_heads += 1
        if torques:
            self.jt_embedding = nn.Sequential(
                nn.Conv2d(1, 1, (3, 1), padding=0), 
                nn.ReLU(),
                nn.Conv2d(1, 1, (2, 1), padding=0), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(joint_state_size,layersize),
                nn.ReLU(),
                nn.Linear(layersize, layersize),
                nn.ReLU()
            )
            n_heads += 1

        self.advantage_stream = torch.nn.Sequential(
            torch.nn.Linear(layersize*n_heads, layersize),
            torch.nn.ReLU(),
            torch.nn.Linear(layersize, n_actions)
        )

        self.value_stream = torch.nn.Sequential(
            torch.nn.Linear(layersize*n_heads, layersize),
            torch.nn.ReLU(),
            torch.nn.Linear(layersize, 1)
        )
        
    def forward(self, myrmex_data, 
                      pose, 
                      j_pos, 
                      j_tau, 
                      j_vel):
        
        embeddings = [self.tactile_ftrs(myrmex_data)]

        if self.pose:
            embeddings.append(self.pose_embedding(pose))
        if self.positions:
            embeddings.append(self.jp_embedding(j_pos))
        if self.velocities:
            embeddings.append(self.jv_embedding(j_vel))
        if self.torques:
            embeddings.append(self.jt_embedding(j_tau))
        
        combined_embedding = torch.cat(embeddings, dim=1)

        advantage = self.advantage_stream(combined_embedding)
        value     = self.value_stream(combined_embedding)
        
        q_values = value + (advantage - advantage.mean())
        
        return q_values

class dueling_placenet_TacTorqPose(nn.Module):
    
    def __init__(self, n_actions, n_timesteps, sensor_type, size) -> None:
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
        
        self.pose_embedding = nn.Sequential(
            nn.Conv2d(1, 16, (5, 1), padding=0, stride=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 1), padding=0, stride=1), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(560,128),
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
    
    def __init__(self, n_actions, n_timesteps, sensor_type, size) -> None:
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

class dueling_placenet_TacRot(nn.Module):
    
    def __init__(self, n_actions, n_timesteps, sensor_type, size) -> None:
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

class dueling_placenet_TacTorque(nn.Module):
    
    def __init__(self, n_actions, n_timesteps, sensor_type, size) -> None:
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
                                        self.jt_embedding(j_tau)
                                        ], dim=1)

        advantage = self.advantage_stream(combined_embedding)
        value     = self.value_stream(combined_embedding)
        
        q_values = value + (advantage - advantage.mean())
        
        return q_values
    
class dueling_placenet_tactileonly(nn.Module):
    
    def __init__(self, n_actions, n_timesteps, sensor_type, size) -> None:
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
                nn.Conv2d(2, 4, (5, 1), padding=0), 
                nn.ReLU(),
                nn.Conv2d(4, 8, (2, 1), padding=0), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(160, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
                )
        

        if sensor_type == 'fingertip':
            self.advantage_stream = torch.nn.Sequential(
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, n_actions)
            )

            self.value_stream = torch.nn.Sequential(
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1)
            )
        else:
            self.advantage_stream = torch.nn.Sequential(
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, n_actions)
            )

            self.value_stream = torch.nn.Sequential(
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1)
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

class dueling_placenet_tacRot(nn.Module):
    
    def __init__(self, n_actions, n_timesteps, sensor_type, size) -> None:
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
                nn.Conv2d(2, 4, (5, 1), padding=0), 
                nn.ReLU(),
                nn.Conv2d(4, 8, (2, 1), padding=0), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(160, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
                )
        
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
        
        tactile_embedding = self.tactile_ftrs(myrmex_data) 
                            
        advantage = self.advantage_stream(tactile_embedding)
        value     = self.value_stream(tactile_embedding)
        
        q_values = value + (advantage - advantage.mean())
        
        return q_values

class dueling_placenet_reduced(nn.Module):
    
    def __init__(self, n_actions, n_timesteps, sensor_type, size) -> None:
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
                nn.Conv2d(2, 1, (3, 1), padding=0), 
                nn.ReLU(),
                nn.Conv2d(1, 1, (3, 1), padding=0), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(72, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
                )
        if sensor_type == 'fingertip':
            self.advantage_stream = torch.nn.Sequential(
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, n_actions)
            )

            self.value_stream = torch.nn.Sequential(
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1)
            )

    def forward(self, myrmex_data):
        
        embedding = self.tactile_ftrs(myrmex_data)

        advantage = self.advantage_stream(embedding)
        value     = self.value_stream(embedding)
        
        q_values = value + (advantage - advantage.mean())
        
class dueling_placenet_reduced(nn.Module):
    
    def __init__(self, n_actions, n_timesteps, sensor_type, size) -> None:
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
                nn.Conv2d(2, 1, (3, 1), padding=0), 
                nn.ReLU(),
                nn.Conv2d(1, 1, (3, 1), padding=0), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(72, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
                )
        
        self.advantage_stream = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_actions)
        )

        self.value_stream = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, myrmex_data):
        
        embedding = self.tactile_ftrs(myrmex_data)

        advantage = self.advantage_stream(embedding)
        value     = self.value_stream(embedding)
        
        q_values = value + (advantage - advantage.mean())
        
        return q_values

class placenet_v2_reduced(nn.Module):

    def __init__(self, n_actions, n_timesteps, sensor_type, size) -> None:
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
                nn.Conv2d(2, 1, (5, 1), padding=0), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(72, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
                )
            

        self.pose_embedding = nn.Sequential(
            nn.Conv2d(1, 1, (5, 1), padding=0), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(42,128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(128*2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_actions)
        )

    def forward(self, myrmex_data, 
                      pose):
        
        combined_embedding = torch.cat([self.tactile_ftrs(myrmex_data), 
                                        self.pose_embedding(pose)
                                       ], dim=1)

        q_values = self.output_net(combined_embedding)
        
        return q_values
    
class placenet_v3_reduced(nn.Module):

    def __init__(self, n_actions, n_timesteps, sensor_type, size) -> None:
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
                nn.Conv2d(2, 1, (5, 1), padding=0), 
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(72, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
                )
            
        self.jt_embedding = nn.Sequential(
                nn.Flatten(start_dim=1, end_dim=2),
                LSTM_custom(7,128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
                )

        if sensor_type == 'fingertip':
            self.output_net = torch.nn.Sequential(
                torch.nn.Linear(128*2, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, n_actions)
            )
        else:
            self.output_net = torch.nn.Sequential(
                torch.nn.Linear(128*2, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, n_actions)
            )


    def forward(self, myrmex_data, 
                      j_tau):
        
        combined_embedding = torch.cat([self.tactile_ftrs(myrmex_data), 
                                        self.jt_embedding(j_tau)
                                       ], dim=1)

        q_values = self.output_net(combined_embedding)
        
        return q_values


class placenet_LSTM_reduced(nn.Module):

    def __init__(self, n_actions, n_timesteps, sensor_type) -> None:
        super().__init__()

        if sensor_type == 'plate':
            self.tactile_ftrs = nn.Sequential(
                Conv2Plus1D(in_channels=2, 
                            out_channels=16,
                            spatial_size=7,
                            temporal_size=1),
                nn.ReLU(),
                Conv2Plus1D(in_channels=16,
                            out_channels=32,
                            spatial_size=5,
                            temporal_size=1),
                nn.ReLU(),
                Conv2Plus1D(in_channels=32,
                            out_channels=1,
                            spatial_size=3,
                            temporal_size=1),
                nn.ReLU(),
                nn.Flatten(start_dim=-2, end_dim=-1),
                nn.Flatten(start_dim=1, end_dim=2),
                LSTM_custom(16, 128), 
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
                )
        elif sensor_type =='fingertip':
            
            self.tactile_ftrs = nn.Sequential(
                nn.Conv2d(2, 1, (1, 1), padding=0), 
                nn.ReLU(),
                nn.Flatten(start_dim=1, end_dim=2),
                LSTM_custom(12,128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
                )
            
        self.pose_embedding = nn.Sequential(
                nn.Flatten(start_dim=1, end_dim=2),
                LSTM_custom(7,128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
                )

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(128*2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_actions)
        )

    def forward(self, myrmex_data, 
                      pose):
 

        combined_embedding = torch.cat([self.tactile_ftrs(myrmex_data), 
                                        self.pose_embedding(pose)
                                       ], dim=1)

        q_values = self.output_net(combined_embedding)
        
        return q_values


if __name__=='__main__':
 
    x  = torch.rand((1, 2, 10, 4))
    x1 = torch.rand((1, 1, 10, 7))
    tactile_ftrs = nn.Sequential(
                nn.Conv2d(2, 8, (5, 1), padding=0), 
                nn.ReLU(),
                nn.Conv2d(8, 16, (2, 1), padding=0, stride=1),
                nn.Flatten()
                )

    jt_embedding = nn.Sequential(
            nn.Conv2d(1, 8, (5, 1), padding=0), 
            nn.ReLU(),
            nn.Conv2d(8, 16, (2, 1), padding=0, stride=1),
            nn.Flatten()
        )
    

    y = tactile_ftrs(x)
    y1=jt_embedding(x1)
    
    c = dueling_placenet(5, 10, 'fingertip')