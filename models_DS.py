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

# dual stream input network 
class Dual_Stream(nn.Module):
    def __init__(self, input_channels, num_conv_layers, num_fc_layers, base_channel_sz, num_classes, input_size, dropout):
        super().__init__()
        
        def make_conv_stream(input_channels):
            layers = []
            current_channels = input_channels
            feature_size = input_size
            
            for i in range(num_conv_layers):
                out_channels = base_channel_sz  # Double the channels for each layer
                layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=1, padding=1))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Reduce spatial size by 2
                
                current_channels = out_channels
                feature_size //= 2  # Update spatial size
                
                # Stop if spatial dimensions become too small
                if feature_size < 2:
                    print(f"Stopping early after {i + 1} layers due to small feature size.")
                    break
            
            return nn.Sequential(*layers), current_channels, feature_size 
        
        def make_fc(fc_input_features, num_classes):
            fc_layers = []
            for i in range(num_fc_layers):
                out_features = fc_input_features // 2
                fc_layers.append(nn.Linear(fc_input_features, out_features))
                fc_layers.append(nn.ReLU())
                fc_layers.append(nn.Dropout(dropout))
                fc_input_features = out_features
            fc_layers.append(nn.Linear(fc_input_features, num_classes))
            return nn.Sequential(*fc_layers)
            
        # Spatial stream
        self.spatial, spatial_out_channels, spatial_feature_size = make_conv_stream(input_channels)
        fc_input_spatial = spatial_out_channels * spatial_feature_size ** 2
        self.fc_spatial = make_fc(fc_input_spatial, num_classes)
        
        # Temporal stream
        temporal_input_channels = 2 * (10 - 1)  # Replace 10 with a dynamic sequence length if needed
        self.temporal_ef, temporal_out_channels, temporal_feature_size = make_conv_stream(temporal_input_channels)
        fc_input_temporal = temporal_out_channels * temporal_feature_size ** 2
        self.fc_temporal = make_fc(fc_input_temporal, num_classes)

    def forward(self, spatial_input, temporal_input):
        # Spatial input: [batch_size, 3, H, W]
        spatial_features = self.spatial(spatial_input)
        spatial_features = spatial_features.view(spatial_features.size(0), -1)  # Flatten
        spatial_logits = self.fc_spatial(spatial_features)
        #spatial_probs = F.softmax(spatial_logits, dim=1)
        
        # Temporal input: [batch_size, 2*(T-1), H, W]
        temporal_features = self.temporal_ef(temporal_input)
        temporal_features = temporal_features.view(temporal_features.size(0), -1)  # Flatten
        temporal_logits = self.fc_temporal(temporal_features)
        #temporal_probs = F.softmax(temporal_logits, dim=1)
        
        # Average the probabilities
        #out = (spatial_probs + temporal_probs) / 2  # Output logits; softmax is applied in loss function
        out = (spatial_logits + temporal_logits) / 2
        return out
    
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