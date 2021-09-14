import torch.nn.functional as F
import torch.nn as nn
import torch

class DuellingDQN(nn.Module):
### Borrowed and adapted from Lab 7 ###
    def __init__(self, input_size, size_hidden, output_size):
        
        super().__init__()
        #2 fully connected layers with ReLu Activation and then action advantage and state value functions
        self.fc1 = nn.Linear(input_size, size_hidden)
        self.bn1 = nn.BatchNorm1d(size_hidden)
        
        self.fc2 = nn.Linear(size_hidden, size_hidden)   
        self.bn2 = nn.BatchNorm1d(size_hidden)

        self.fc3 = nn.Linear(size_hidden, size_hidden)  
        
        self.fc_value = nn.Linear(size_hidden, 1)
        
        self.fc_adv = nn.Linear(size_hidden,output_size)
        
    def forward(self, x):
        
        h1 = F.relu(self.bn1(self.fc1(x.float())))
        h2 = F.relu(self.bn2(self.fc2(h1)))
       # h3 = F.relu(self.bn3(self.fc3(h2)))
        o = self.fc3(h2)
        
        adv = self.fc_adv(o)
        
        val = self.fc_value(o).expand(-1, 4)
        
        output = val + adv - torch.mean(adv, dim=1, keepdim=True) #take the mean of advantage from the sum
        
        return output