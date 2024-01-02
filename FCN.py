import torch
from torch import nn

class MNISTClassifier(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(MNISTClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, output_size),
        )
        
    def forward(self, x):
        return self.model(x)
    
