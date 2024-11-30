from datasets import FrameImageDataset, get_transforms, FrameVideoDataset
from models import Base_Network, build_optimizer, save_checkpoint
from trainers import _train_every_frame
from datasets import FrameImageDataset, get_transforms, FrameVideoDataset

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

import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import torch

# Your normalization parameters (mean, std)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def unnormalize(tensor_image):
    """
    Unnormalize a tensor image (from [-1, 1] range to original range based on given mean and std).
    tensor_image is expected to be in CxHxW format (C = channels, H = height, W = width).
    """
    for i in range(3):  # Assuming RGB image (3 channels)
        tensor_image[i, :, :] = tensor_image[i, :, :] * std[i] + mean[i]
    
    return tensor_image


if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

early_fusion = False
every_frame = True

root_dir = "./data/ufc10"

if early_fusion:# if config.network == "resnet18":
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.conv1 = nn.Conv2d(in_channels=30, out_channels=64, kernel_size=(7,7), padding=(3,3), stride=(2,2), bias=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)
    PATH = "/work3/s164248/data/trained_models/checkpoint_2730bb3l.pth"
    data_transforms = get_transforms(rotation_degree = 30, transform_size = 224, video_consistency= True)
    framevideostack_dataset_test = FrameVideoDataset(root_dir=root_dir, split='test', transform=data_transforms['test'], stack_frames = True, clara_insisted=True)
    TEST_SET = DataLoader(framevideostack_dataset_test,  batch_size=1, shuffle=False)
elif every_frame:
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    num_classes = 10  
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    PATH = "/work3/s164248/data/trained_models/checkpoint_hn4ut59q.pth"
    data_transforms = get_transforms(rotation_degree = 30, transform_size = 224)
    frameimage_dataset_test = FrameImageDataset(root_dir=root_dir, split='test', transform=data_transforms['test'])
    TEST_SET = DataLoader(frameimage_dataset_test,  batch_size=1, shuffle=False)


# Load the checkpoint with device mapping
checkpoint = torch.load(PATH, map_location=device)  # Map to the appropriate device

# Load the model state_dict
model.load_state_dict(checkpoint['model_state_dict'])

# Move the model to the appropriate device
model.to(device)

# Set the model to evaluation mode
model.eval()


import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# Counter to track the number of images processed
image_counter = 0
total_counter = 0
# Number of images per subplot (in a single row or column)
images_per_subplot = 10

# Creating a new figure for every 10 images
fig, axes = plt.subplots(1, images_per_subplot, figsize=(15, 10))  # Adjust figsize for 10 images
predicted_item_list=[]
output_item_list=[]
with torch.no_grad():  # Disable gradient computation for evaluation
    for data, target in TEST_SET:
        # Move data and target to the specified device
        data, target = data.to(device), target.to(device).long()
        
        # Forward pass
        output = model(data)

        # Compute predictions
        predicted = torch.argmax(output, dim=1)  # Multi-class prediction

        # Process the single image in the batch (since batch size is 1)
        frame = data.squeeze(0).cpu()  # Get the individual frame, remove batch dimension
        plot_image = True
        
        if plot_image:
            frames = frame.numpy()  # Convert frame to numpy array (shape: 3, 224, 224)

            # Unnormalize the frame
            unnormalized_frame = unnormalize(torch.tensor(frames))

            # Plot the image
            axes[image_counter].imshow(unnormalized_frame.permute(1, 2, 0).cpu().numpy())  # Convert (3, 224, 224) to (224, 224, 3)
            axes[image_counter].axis('off')  # Hide axes
            
            # Optionally set titles or any other text
            axes[image_counter].set_title(f"Label: {target.item()}, Pred: {predicted.item()}", fontsize=10)
            predicted_item_list.append(predicted.item())
            output_item_list.append(output)
            # Increment the counter for the subplot position
            image_counter += 1
            total_counter += 1
            # If we've plotted 10 images, save and reset
            if image_counter == images_per_subplot:
                pred = torch.argmax(sum(output_item_list,1),dim=1)
                plt.suptitle(f"Images {image_counter - 9} to {image_counter} | prediction: {pred}", fontsize=16)
                plt.tight_layout()
                plt.savefig(f"./images/test_images_{total_counter // images_per_subplot}.png")
                plt.close(fig)  # Close the current figure
                fig, axes = plt.subplots(1, images_per_subplot, figsize=(15, 10))  # New figure for the next 10 images
                image_counter = 0  # Reset the counter for the next set of 10 images
                
                predicted_item_list=[]
                output_item_list=[]
                
                target.item()


                 

# If there are remaining images after the loop
if image_counter > 0:
    plt.suptitle(f"Final set of {image_counter} Images", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"./images/test_images_final.png")
    plt.close(fig)



print('hello')