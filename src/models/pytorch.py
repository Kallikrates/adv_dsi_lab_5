import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class PytorchRegression(nn.Module):
    def __init__(self, num_features):
        super(PytorchRegression, self).__init__()
        
        self.layer_1 = nn.Linear(num_features, 128)
        self.layer_out = nn.Linear(128, 1)

    def forward(self, x):
        x = F.dropout(F.relu(self.layer_1(x)))
        x = self.layer_out(x)
        return (x)
        

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU 
    return device
    

class PytorchDataset(Dataset):
    """
    Pytorch dataset
    ...

    Attributes
    ----------
    X_tensor : Pytorch tensor
        Features tensor
    y_tensor : Pytorch tensor
        Target tensor

    Methods
    -------
    __getitem__(index)
        Return features and target for a given index
    __len__
        Return the number of observations
    to_tensor(data)
        Convert Pandas Series to Pytorch tensor
    """
        
    def __init__(self, X, y):
        self.X_tensor = self.to_tensor(X)
        self.y_tensor = self.to_tensor(y)
    
    def __getitem__(self, index):
        return self.X_tensor[index], self.y_tensor[index]
        
    def __len__ (self):
        return len(self.X_tensor)
        
class PytorchBinary(nn.Module):
    def __init__(self, num_features):
        super(PytorchBinary, self).__init__()
        
        self.layer_1 = nn.Linear(num_features, 256)
        self.layer_out = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.dropout(F.relu(self.layer_1(x)))
        x = self.sigmoid(self.layer_out(x))
        return (x)
    
    def to_tensor(self, data):
        return torch.Tensor(np.array(data))
