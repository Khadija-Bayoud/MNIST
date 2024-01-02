import torch
from torch import nn

class CONVClassifier(nn.Module):
    def __init__(self):
        super(CONVClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2), 
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2), 
            nn.Flatten(),
            nn.Linear(256, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
        )
        
    def forward(self, x):
        return self.model(x)
    
