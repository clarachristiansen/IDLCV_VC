from datasets import get_transforms, FrameVideoDataset
from models import build_optimizer
from trainers import _train_every_frame

from torch.utils.data import DataLoader
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
        'value': 100 ### change for true training
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
        "values": ["resnet_3d"]#["base_network", "resnet_3d"]
    }
}
sweep_config['metric'] = metric
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project='Video_early_initial')

import torch
from matplotlib import pyplot as plt

criterion = nn.CrossEntropyLoss()

def run_wandb(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config
        run_id = wandb.run.id 
        config.run_id = run_id
        wandb.run.name = f"Run {run_id}"
        
        data_transforms = get_transforms(rotation_degree = 30, transform_size = config.image_size, video_consistency= True)

        ''' Load data '''  
        framevideostack_dataset_train = FrameVideoDataset(root_dir=root_dir, split='train', transform=data_transforms['train'],stack_frames = True, clara_insisted=False)
        framevideostack_dataset_val = FrameVideoDataset(root_dir=root_dir, split='val', transform=data_transforms['val'], stack_frames = True, clara_insisted=False)
        framevideostack_dataset_test = FrameVideoDataset(root_dir=root_dir, split='test', transform=data_transforms['test'], stack_frames = True, clara_insisted=False)

        framevideostack_loader_train = DataLoader(framevideostack_dataset_train,  batch_size=config.batch_size, shuffle=True)
        framevideostack_loader_val = DataLoader(framevideostack_dataset_val,  batch_size=1, shuffle=False)
        framevideostack_loader_test = DataLoader(framevideostack_dataset_test,  batch_size=1, shuffle=False)

        # if config.network == "resnet_3d":
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True).to(device)

        # Replace the classification head with a custom output for 10 classes
        model.blocks[5].proj = nn.Linear(model.blocks[5].proj.in_features, 10).to(device)
        model.to(device)
        
        optimizer = build_optimizer(model, config.learning_rate)

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
    root_dir = './dtu/datasets1/02516/ucf101_noleakage'

    wandb.agent(sweep_id, run_wandb)
