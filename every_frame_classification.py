from datasets_new import FrameImageDataset, get_transforms, DualStreamDataset
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
    'epochs': {'value': 10},
    'learning_rate': {"values": [0.001, 0.01]},
    'batch_size': {"values": [ 32, 64]},
    'image_size': {'value': 224},
    'num_conv_layers': {"values": [2, 3, 4]},  
    'num_fc_layers': {"values": [1, 2, 3]},    
    'dropout': {"values": [0.2, 0.0]},         
    "network": {"values": ["dual_stream"]}
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
        
        data_transforms = get_transforms(rotation_degree=30, transform_size=config.image_size)

        if config.network == "dual_stream":
            # Load DualStreamDataset for spatial and temporal inputs
            dualstream_dataset_train = DualStreamDataset(
                root_dir=root_dir, 
                split='train', 
                transform=data_transforms['train'], 
                resize=(config.image_size, config.image_size), 
                n_sampled_frames=10
            )
            dualstream_dataset_val = DualStreamDataset(
                root_dir=root_dir, 
                split='val', 
                transform=data_transforms['val'], 
                resize=(config.image_size, config.image_size), 
                n_sampled_frames=10
            )
            dualstream_dataset_test = DualStreamDataset(
                root_dir=root_dir, 
                split='test', 
                transform=data_transforms['test'], 
                resize=(config.image_size, config.image_size), 
                n_sampled_frames=10
            )

            dualstream_loader_train = DataLoader(dualstream_dataset_train, batch_size=config.batch_size, shuffle=True)
            dualstream_loader_val = DataLoader(dualstream_dataset_val, batch_size=1, shuffle=False)
            dualstream_loader_test = DataLoader(dualstream_dataset_test, batch_size=1, shuffle=False)

            # Initialize Dual_Stream model
            model = Dual_Stream(
                input_channels=3,  # RGB for spatial input
                num_conv_layers=config.num_conv_layers,
                num_fc_layers=config.num_fc_layers,
                base_channel_sz=64,
                num_classes=10,
                input_size=config.image_size,
                dropout=config.dropout
            )
            train_loader = dualstream_loader_train
            val_loader = dualstream_loader_val
            test_loader = dualstream_loader_test
            train_dataset = dualstream_dataset_train
            val_dataset = dualstream_dataset_val
            test_dataset = dualstream_dataset_test
        else:
            # Load FrameImageDataset for single-stream networks
            frameimage_dataset_train = FrameImageDataset(root_dir=root_dir, split='train', transform=data_transforms['train'])
            frameimage_dataset_val = FrameImageDataset(root_dir=root_dir, split='val', transform=data_transforms['val'])
            frameimage_dataset_test = FrameImageDataset(root_dir=root_dir, split='test', transform=data_transforms['test'])

            frameimage_loader_train = DataLoader(frameimage_dataset_train, batch_size=config.batch_size, shuffle=True)
            frameimage_loader_val = DataLoader(frameimage_dataset_val, batch_size=1, shuffle=False)
            frameimage_loader_test = DataLoader(frameimage_dataset_test, batch_size=1, shuffle=False)

            if config.network == "resnet18":
                model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
                num_classes = 10  # Your custom number of output classes
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif config.network == "base_network":
                model = Base_Network(config.dropout, config.num_layers, num_classes=10)

            train_loader = frameimage_loader_train
            val_loader = frameimage_loader_val
            test_loader = frameimage_loader_test
            train_dataset = frameimage_dataset_train
            val_dataset = frameimage_dataset_val
            test_dataset = frameimage_dataset_test

        model.to(device)
        optimizer = build_optimizer(model, config.learning_rate)

        # Train and evaluate the model
        _train_every_frame(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            num_epochs=config.epochs,
            run_id=config.run_id
        )
        
if __name__ == '__main__':
    ''' Standard configurations '''
    root_dir = './data/ufc10/'

    wandb.agent(sweep_id, run_wandb)