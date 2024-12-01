# dataset2.py
from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T
from custom_transforms import *

import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set a fixed seed
set_seed(67120349)

class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir = '/work3/ppar/data/ucf101', 
    split = 'train', 
    transform = None,
    stack_frames = True,
    clara_insisted = False
):

        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        self.clara_insisted = clara_insisted
        
        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)
    
    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split('/')[-1].split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        video_frames_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'frames')
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            # Apply consistent transformations to all frames at once
            try:
                frames = self.transform(video_frames)
            except: 
                frames = [self.transform(frame) for frame in video_frames]
            if not isinstance(frames, list):
                frames = [frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]
        
        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)
        
        if self.clara_insisted:
            frames = frames.reshape([30,224,224])

        return frames, label
    
    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames

def get_transforms(rotation_degree=30, transform_size=128, video_consistency=False):
    normalize_array = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    if video_consistency:
        video_transform = T.Compose([
            ConsistentRandomRotation(rotation_degree),
            ConsistentRandomResizedCrop(transform_size),
            ConsistentRandomHorizontalFlip(),
            ConsistentColorJitter(),
            T.Lambda(lambda frames: [T.ToTensor()(frame) for frame in frames]),
            T.Lambda(lambda frames: [T.Normalize(normalize_array[0], normalize_array[1])(frame) 
                                   for frame in frames])
        ])
    else:
        video_transform = T.Compose([
            T.Resize(transform_size),
            T.CenterCrop(transform_size),
            T.ColorJitter(),
            T.ToTensor(),
            T.Normalize(normalize_array[0], normalize_array[1])
        ])

    basic_trans = T.Compose([
        T.RandomRotation(rotation_degree),
        T.RandomResizedCrop(transform_size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(),
        T.ToTensor(),
        T.Normalize(normalize_array[0], normalize_array[1])
    ])

    data_transforms = {
        'train': video_transform if video_consistency else basic_trans,
        'val': T.Compose([
            T.Resize(transform_size),
            T.CenterCrop(transform_size),
            T.ToTensor(),
            T.Normalize(normalize_array[0], normalize_array[1])
        ]),
        'test': T.Compose([
            T.Resize(transform_size),
            T.CenterCrop(transform_size),
            T.ToTensor(),
            T.Normalize(normalize_array[0], normalize_array[1])
        ])
    }
    return data_transforms

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    root_dir = './data/ufc10/'

    transform = T.Compose([T.Resize((64, 64)),T.ToTensor()])
    frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
    framevideostack_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = True)
    framevideolist_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = False)
    
    # frameimage_loader = DataLoader(frameimage_dataset,  batch_size=8, shuffle=False)
    framevideostack_loader = DataLoader(framevideostack_dataset,  batch_size=8, shuffle=False)
    framevideolist_loader = DataLoader(framevideolist_dataset,  batch_size=8, shuffle=False)

    # for frames, labels in frameimage_loader:
    #     print(frames.shape, labels, labels.shape) # [batch, channels, height, width]

    for video_frames, labels in framevideolist_loader:
        print(45*'-')
        for frame in video_frames: # loop through number of frames
            print(frame.shape, labels.shape)# [batch, channels, height, width]

    # for video_frames, labels in framevideostack_loader:
    #     print(video_frames.shape, labels.shape) # [batch, channels, number of frames, height, width]
            
