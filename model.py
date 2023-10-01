import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.v1 = nn.Linear(input_size, hidden_size) 
        self.v2 = nn.Linear(hidden_size, hidden_size) 
        self.v3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.v1(x)
        out = self.relu(out)
        out = self.v2(out)
        out = self.relu(out)
        out = self.v3(out)
        return out