from datasets import FrameImageDataset, get_transforms, FrameVideoDataset
from models import Base_Network, build_optimizer, save_checkpoint
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
    'name': 'validation_loss',
    'goal': 'minimize'
  }

parameters_dict = {
    # 'dropout' : {
    #     'values' : [0.2, 0.5]
    # },
    'epochs' : {
        'value': 200 ### change for true training
    },
    'learning_rate' : {
        "values": [0.01]#, 0.001]
    },
    'batch_size': {
        "values": [32,64]
    },
    'image_size':{
        'value': 224
    },
}
early_stopping ={
        "type": "hyperband",
        "eta": 3,
        "min_iter":30
     }
sweep_config['metric'] = metric
sweep_config['parameters'] = parameters_dict
sweep_config['early_terminate'] = early_stopping

sweep_id = wandb.sweep(sweep_config, project='Video_early_noleakage_try4')

import torch
from matplotlib import pyplot as plt

criterion = nn.CrossEntropyLoss()

def run_wandb(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config, dir="/work3/s164248/data/wandb"): #this has been hardcoded, sorry..
        config = wandb.config
        run_id = wandb.run.id 
        config.run_id = run_id
        wandb.run.name = f"Run {run_id}"
        
        data_transforms = get_transforms(rotation_degree = 30, transform_size = config.image_size, video_consistency= True)

        ''' Load data '''  
        framevideostack_dataset_train = FrameVideoDataset(root_dir=root_dir, split='train', transform=data_transforms['train'],stack_frames = True, clara_insisted=True)
        framevideostack_dataset_val = FrameVideoDataset(root_dir=root_dir, split='val', transform=data_transforms['val'], stack_frames = True, clara_insisted=True)
        framevideostack_dataset_test = FrameVideoDataset(root_dir=root_dir, split='test', transform=data_transforms['test'], stack_frames = True, clara_insisted=True)

        framevideostack_loader_train = DataLoader(framevideostack_dataset_train,  batch_size=config.batch_size, shuffle=True)
        framevideostack_loader_val = DataLoader(framevideostack_dataset_val,  batch_size=1, shuffle=False)
        framevideostack_loader_test = DataLoader(framevideostack_dataset_test,  batch_size=1, shuffle=False)
        

        # if config.network == "resnet18":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)


        # trying here with relu
        model.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=64, kernel_size=(7,7), padding=(3,3), stride=(2,2), bias=False),
            # nn.ReLU()  
        )

        # model.conv1 = nn.Conv2d(in_channels=30, out_channels=64, kernel_size=(3,3), padding=(3,3), stride=(2,2), bias=False)
        #trying with dropout in last fc
        model.fc = nn.Sequential(                         # Apply dropout here
            nn.Linear(model.fc.in_features, 10)                     # Final classification layer
        )
        model.to(device)

        optimizer = build_optimizer(model, config.learning_rate, weight_decay=1e-1)

        # # Generate a random id for this run and this model
        _train_every_frame(model, optimizer, criterion, 
                            framevideostack_loader_train,
                            framevideostack_loader_val,
                            framevideostack_loader_test,
                            framevideostack_dataset_train, 
                            framevideostack_dataset_val, 
                            framevideostack_dataset_test, 
                            num_epochs=config.epochs, 
                            run_id=config.run_id)
        
if __name__ == '__main__':
    ''' Standard configurations '''
    root_dir = '/dtu/datasets1/02516/ucf101_noleakage'
    # root_dir = '/zhome/88/7/117159/Courses/IDLCV_VC/data/ufc10/ufc10_no_leakage'

    wandb.agent(sweep_id, run_wandb)
