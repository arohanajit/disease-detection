import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torchvision.transforms as transforms

class malaria_model(nn.Module):
    
    def __init__(self):
        super(malaria_model, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.2)
        )
        
            
        self.fc1 = nn.Linear(32*4*4, 512)
        self.batch_norm1 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 2)
        
    def forward(self, x):
        out = self.block(x)
        out = self.block2(out)
        out = self.block2(out)
        out = out.view(out.size(0), -1)   # flatten out a input for Dense Layer
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class breast_model(nn.Module):

    def __init__(self):
        super(breast_model, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.2)
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.2)
        )

        self.fc1 = nn.Linear(64*3*3, 512)
        self.pool = nn.MaxPool2d(2)
        self.batch_norm = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        out = self.block(x)
        out = self.block2(out)
        out = self.conv_block(out)
        out = self.block3(out)
        out = out.view(out.size(0), -1)   # flatten out a input for Dense Layer
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

class glaucoma_model(nn.Module):
    def __init__(self):
        super(glaucoma_model, self).__init__()
        self.model = models.densenet121(pretrained=True)
        self.model.classifier = nn.Sequential(
                nn.Linear(1024, 1024, bias=True),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 2, bias=True)
            )

    def forward(self,x):
        out = self.model(x)
        return out



