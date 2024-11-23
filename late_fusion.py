import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from datasets import FrameVideoDataset
from torchvision import transforms as T
import os
from tqdm import tqdm 
import torch.optim as optim


os.chdir('./IDLCV_VC')
print(os.getcwd())
def get_transforms(rotation_degree = 30, transform_size = 128 ):
    normalize_array= ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    data_transforms = {
    'train': T.Compose([
        T.RandomRotation(rotation_degree),
        T.RandomResizedCrop(transform_size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(),
        T.ToTensor(),
        T.Normalize(normalize_array[0], normalize_array[1])
    ]),
    'val': T.Compose([
        T.Resize(transform_size),
        T.CenterCrop(transform_size),
        T.ToTensor(),
        T.Normalize(normalize_array[0], normalize_array[1])
         ]),
    'test':
    T.Compose([
        T.Resize(transform_size),
        T.CenterCrop(transform_size),
        T.ToTensor(),
        T.Normalize(normalize_array[0], normalize_array[1])
    ])
    }
    return data_transforms

data_transforms = get_transforms(30, 128)
FV_dataset = FrameVideoDataset(split = 'train', transform=data_transforms['train'])

FV_dataloader = DataLoader(FV_dataset,  batch_size=8, shuffle=False)

#late_fusion_models = [torch.nn.Sequential(*(list(models.resnet18(weights='IMAGENET1K_V1').children())[:-2]))  for _ in range(10)]

#for video_frames, labels in FV_dataloader:
#        features = late_fusion_models[0](video_frames[:,:,0,:,:]).shape
#        print(45*'-')


class LateFusionModel(nn.Module):
    def __init__(self, num_models=10):
        super(LateFusionModel, self).__init__()
        self.models = nn.ModuleList([
            nn.Sequential(*(list(models.resnet18(weights='IMAGENET1K_V1').children())[:-2])) for _ in range(num_models)
        ])
        # Add a classifier on top of the concatenated features from all models
        self.fc = nn.Linear(512 * num_models * 4 * 4, 10)  # Assuming 10 classes for classification

    def forward(self, x):
        # Get the features from each model and aggregate
        features = [model(x[:,:,i,:,:]) for i, model in enumerate(self.models)]  # x[:,:,i,:,:] for each model
        aggregated_features = torch.cat(features, dim=1)  # Concatenate features from all models
        flatten_features = aggregated_features.reshape((aggregated_features.shape[0], aggregated_features.shape[1]*4*4))
        out = self.fc(flatten_features)  # Final classification layer
        return out

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LateFusionModel(num_models=10).to(device)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize performance tracking
out_dict = {
    'train_acc': [],
    'validation_acc': [],
    'test_acc': [],
    'train_loss': [],
    'validation_loss': [],
}
previous_val_acc = 1000000
num_epochs = 100

# Training loop
for epoch in tqdm(range(num_epochs), unit='epoch'):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    # Loop through batches of data
    for video_frames, labels in FV_dataloader:
        # Move data and target to the device
        video_frames, labels = video_frames.to(device), labels.to(device)

        # Forward pass: pass the video frames through the model
        outputs = model(video_frames)

