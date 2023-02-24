import numpy as np
import torch.nn as nn
import torch as T
import torch.optim as optim
import torch.nn.functional as F
import os

class DDPGActor(nn.Module):
    def __init__(self, input_dims: int = None, output_dims: int = None, name : str = '',chkpt : str = '', beta : float=1e-04 ):
        super().__init__()
        print(output_dims)        
        self.chkpt = os.path.join(chkpt, 'runs', name)        
        self.network = nn.Sequential(
            nn.Linear(in_features=input_dims, out_features=128, bias=True),
            nn.Dropout(p=0.5),
            # nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64, bias=True),
            nn.Dropout(p=0.5),
            # nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=output_dims, bias=True),
            nn.Dropout(p=0.5),
            # nn.BatchNorm1d(num_features=64),
            nn.ReLU())        
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)
    def forward(self, x):
        x = self.network(x)
        return x

    def save(self):
        T.save(self.state_dict(), self.chkpt)

    def load(self):
        self.load_state_dict(T.load(self.chkpt))

class DDPGCritic(nn.Module):
    def __init__(self, input_dims : any, beta :float = 1e-05, chkpt : str = '', name: str = ''):
        super().__init__()
        self.chkpt = os.path.join(chkpt, 'runs', name)       
        self.network = nn.Sequential(
            nn.Linear(in_features=input_dims, out_features=128, bias=True),
            nn.Dropout(p=0.5),
            # nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64, bias=True),
            nn.Dropout(p=0.5),
            # nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1, bias=True),
            nn.Dropout(p=0.5),
            # nn.BatchNorm1d(),
            nn.ReLU())
       

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)        

    def forward(self, state, actions):
        tnsr = T.cat([state, actions], dim=2)
        s = tnsr.shape
        tnsr = tnsr.reshape((s[0], s[1] * s[2]))
        x = self.network(tnsr)
        return x

    def save(self):
        T.save(self.state_dict(),self.chkpt)

    def load(self):
        self.load_state_dict(T.load(self.chkpt))

if __name__ == '__main__':

    x = DDPGActor(10, 2)
    print(x(T.randn(1, 10)))