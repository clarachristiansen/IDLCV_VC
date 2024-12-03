import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from datasets import FrameVideoDataset, get_transforms
from torchvision import transforms as T
import os
from tqdm import tqdm 
import torch.optim as optim
from IDLCV_VC.trainers_DS import _train_every_frame
import wandb

wandb.login(key='4aaf96e30165bfe476963bc860d96770512c8060')

#os.chdir('./IDLCV_VC')
print(os.getcwd())

class LateFusionModel(nn.Module):
    def __init__(self, num_models=10):
        super(LateFusionModel, self).__init__()
        self.models = nn.ModuleList([
            nn.Sequential(*(list(models.resnet18(pretrained=True).children())[:-1])) for _ in range(num_models)
        ])
        self.fc = nn.Sequential(nn.Linear(512 * num_models, 128),
                                nn.ReLU(),
                                nn.Linear(128, 10)) 
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        features = [model(x[:,:,i,:,:]) for i, model in enumerate(self.models)]  # x[:,:,i,:,:] for each model
        aggregated_features = torch.cat(features, dim=1)  # Concatenate features from all models
        flatten_features = aggregated_features.reshape((aggregated_features.shape[0], aggregated_features.shape[1]))
        out = self.fc(flatten_features)  # Final classification layer
        #out = self.softmax(out)
        return out

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)
model = LateFusionModel(num_models=10).to(device)

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
        'value': 100 ### change for true training
    },
    'learning_rate' : {
        "values": [0.01, 0.001]
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
}
sweep_config['metric'] = metric
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project='Video_late_no_leakage')

import torch
from matplotlib import pyplot as plt

criterion = nn.CrossEntropyLoss()
root_dir = '/dtu/datasets1/02516/ucf101_noleakage' #'./data/ufc10' # has leakage

def run_wandb(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config
        run_id = wandb.run.id 
        config.run_id = run_id
        wandb.run.name = f"Run {run_id}"
        
        data_transforms = get_transforms(rotation_degree = 30, transform_size = config.image_size, video_consistency=True)

        ''' Load data '''  
        FV_dataset_train = FrameVideoDataset(root_dir=root_dir, split = 'train', transform=data_transforms['train'])
        FV_dataloader_train = DataLoader(FV_dataset_train,  batch_size=config.batch_size, shuffle=True)

        FV_dataset_val = FrameVideoDataset(root_dir=root_dir, split = 'val', transform=data_transforms['val'])
        FV_dataloader_val = DataLoader(FV_dataset_val,  batch_size=1, shuffle=False)

        FV_dataset_test = FrameVideoDataset(root_dir=root_dir, split = 'test', transform=data_transforms['test'])
        FV_dataloader_test = DataLoader(FV_dataset_test,  batch_size=1, shuffle=False)


        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        # # Generate a random id for this run and this model
        _train_every_frame(model, optimizer, criterion, 
                            FV_dataloader_train,
                            FV_dataloader_val,
                            FV_dataloader_test,
                            None, 
                            None, 
                            None, 
                            num_epochs=config.epochs, 
                            run_id=config.run_id)
        
if __name__ == '__main__':
    wandb.agent(sweep_id, run_wandb)