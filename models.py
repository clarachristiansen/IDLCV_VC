import os
import numpy as np
import glob
import PIL.Image as Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import wandb

class Base_Network(nn.Module):
    def __init__(self, dropout, num_layers, feature_size=128, base_channel_sz = 8, num_classes= 10):
        super(Base_Network, self).__init__()

        self.beginning = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=base_channel_sz, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(base_channel_sz),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        feature_size //= 2 # Convolution -> Padding -> Pooling

        layers = []
        for i in range(num_layers):
          layers.append(nn.Conv2d(in_channels=base_channel_sz*(i+1), out_channels=base_channel_sz*(i+2), kernel_size=3, padding=1))
          layers.append(nn.BatchNorm2d(base_channel_sz*(i+2)))
          layers.append(nn.ReLU())
          
          layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
          feature_size //= 2 # Convolution -> Padding -> Pooling

        self.convolutional = nn.Sequential(*layers)
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features= base_channel_sz * (num_layers + 1) * feature_size * feature_size, out_features=512),
            nn.ReLU(),
            nn.Dropout(dropout), # added dropout to reduce overfitting

            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.beginning(x)
        x = self.convolutional(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)

        x = self.fully_connected(x)
        return x


def build_optimizer(model, learning_rate):
    ''' lets just select one, as we dont really see a performance difference between sgd and adam.. adam is usually better.'''
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer

def save_checkpoint(model, optimizer, epoch, path='./checkpoints/model_checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, path)
    print(f"Model checkpoint saved at epoch {epoch}, name: {path}")