from datasets_new import FrameImageDataset, get_transforms
from models import Base_Network, build_optimizer, save_checkpoint, Dual_Stream
from trainers import _train_every_frame

from torch.utils.data import DataLoader
from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
import wandb
wandb.login(key='4aaf96e30165bfe476963bc860d96770512c8060')

import torch
import torch.nn as nn

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


sweep_config = {
    'method' : "bayes", # 'random', #bayes
}

metric = {
    'name': 'loss',
    'goal': 'minimize'
  }

parameters_dict = {
    # 'dropout' : {
    #     'values' : [0.2, 0.5]
    # },
    'epochs' : {
        'value': 10 ### change for true training
    },
    'learning_rate' : {
        "values": [0.001, 0.01]
    },
    'batch_size': {
        "values": [8, 16, 32, 64]
    },
    'image_size':{
        'value': 224
    },
    # "num_layers": {
    #     "values": list(range(4, 6))
    # },
    "network": {
        "values": ["resnet18"]#["base_network", "resnet18"]
    }
}
sweep_config['metric'] = metric
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project='Video_initial_resnet')

import torch



criterion = nn.CrossEntropyLoss()

def run_wandb(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config
        run_id = wandb.run.id 
        config.run_id = run_id
        wandb.run.name = f"Run {run_id}"
        
        data_transforms = get_transforms(rotation_degree = 30, transform_size = config.image_size)

        ''' Load data '''  
        frameimage_dataset_train = FrameImageDataset(root_dir=root_dir, split='train', transform=data_transforms['train'])
        frameimage_dataset_val = FrameImageDataset(root_dir=root_dir, split='val', transform=data_transforms['val'])
        frameimage_dataset_test = FrameImageDataset(root_dir=root_dir, split='test', transform=data_transforms['test'])

        frameimage_loader_train = DataLoader(frameimage_dataset_train,  batch_size=config.batch_size, shuffle=True)
        frameimage_loader_val = DataLoader(frameimage_dataset_val,  batch_size=1, shuffle=False)
        frameimage_loader_test = DataLoader(frameimage_dataset_test,  batch_size=1, shuffle=False)
        
        if config.network == "resnet18":
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
            num_classes = 10  # Your custom number of output classes
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif config.network == "base_network":
            model = Base_Network(config.dropout, config.num_layers, num_classes=10) 
        elif config.network == "dual_stream":
            model = Dual_Stream(
            input_channels=3,  # Assuming RGB images for spatial input
            num_conv_layers=config.num_conv_layers,  # Number of convolutional layers
            num_fc_layers=config.num_fc_layers,  # Number of fully connected layers
            base_channel_sz=64,  # Base channel size; adjust as needed
            num_classes=10,  # Number of output classes
            input_size=config.image_size,  # Image size (H, W)
            dropout=config.dropout)  # Dropout rate
        model.to(device)
    
        

        optimizer = build_optimizer(model, config.learning_rate)

        # Generate a random id for this run and this model
        _train_every_frame(model, optimizer, criterion, 
                            frameimage_loader_train,
                            frameimage_loader_val,
                            frameimage_loader_test,
                            frameimage_dataset_train, 
                            frameimage_dataset_val, 
                            frameimage_dataset_test, 
                            num_epochs=config.epochs, 
                            run_id=config.run_id)
        
if __name__ == '__main__':
    ''' Standard configurations '''
    root_dir = './data/ufc10/'

    wandb.agent(sweep_id, run_wandb)
