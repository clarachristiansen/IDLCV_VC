from glob import glob
import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T

import torch.nn.functional as F

class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir='/work3/ppar/data/ucf101_noleakage',
    split='train', 
    transform=None
):
        self.frame_paths = sorted(glob(f'{root_dir}/frames/{split}/*/*/*.jpg'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        # breakpoint()
       
    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        
        video_name = frame_path.split('/')[-2]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()
        
        frame = Image.open(frame_path).convert("RGB")

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)

        return frame, label


class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir = '/work3/ppar/data/ucf101_noleakage', 
    split = 'train', 
    transform = None,
    stack_frames = True
):

        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        
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
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]
        
        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)


        return frames, label
    
    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames


class FlowVideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir = '/work3/ppar/data/ucf101_noleakage', 
    split = 'train', 
    resize = (64,64),
):

        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.resize = resize    
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

        video_flows_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'flows')
        flows = self.load_flows(video_flows_dir)

        return flows, label

    def load_flows(self, flows_dir):
        flows = []
        for i in range(1, self.n_sampled_frames):
            flow_file = os.path.join(f'{flows_dir}', f"flow_{i}_{i+1}.npy")
            flow = np.load(flow_file)
            flow = torch.from_numpy(flow)
            flows.append(flow)
        flows = torch.stack(flows)

        if self.resize:
            flows = F.interpolate(flows, size=self.resize, mode='bilinear')

        return flows.flatten(0,1)
   
   
    
class DualStreamDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir='/work3/ppar/data/ucf101_noleakage', split='train', transform=None, resize=(64, 64), n_sampled_frames=10):
        self.frame_dataset = FrameVideoDataset(root_dir=root_dir, split=split, transform=transform, stack_frames=False)
        self.flow_dataset = FlowVideoDataset(root_dir=root_dir, split=split, resize=resize)
        self.n_sampled_frames = n_sampled_frames

    def __len__(self):
        return min(len(self.frame_dataset), len(self.flow_dataset))

    def __getitem__(self, idx):
        video_frames, label = self.frame_dataset[idx]
        if len(video_frames) < self.n_sampled_frames:
            raise ValueError(f"Video at index {idx} has fewer than {self.n_sampled_frames} frames.")
        video_frames = video_frames[:self.n_sampled_frames]
        spatial_input = torch.stack(video_frames).permute(1, 0, 2, 3)
        temporal_input, flow_label = self.flow_dataset[idx]
        assert label == flow_label, f"Labels do not match for index {idx}: {label} != {flow_label}"
        return spatial_input, temporal_input, label


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    root_dir = '/work3/ppar/data/ucf101_noleakage'

    transform = T.Compose([T.Resize((64, 64)),T.ToTensor()])
    frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
    framevideostack_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = True)
    framevideolist_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = False)
    flowvideo_dataset = FlowVideoDataset(root_dir=root_dir, split='val', resize=(64,64))
    dualstream_dataset = DualStreamDataset(
        root_dir=root_dir, 
        split='val', 
        transform=transform, 
        resize=(64, 64), 
        n_sampled_frames=10
    )


    frameimage_loader = DataLoader(frameimage_dataset,  batch_size=8, shuffle=False)
    framevideostack_loader = DataLoader(framevideostack_dataset,  batch_size=8, shuffle=False)
    framevideolist_loader = DataLoader(framevideolist_dataset,  batch_size=8, shuffle=False)
    flowvideo_loader = DataLoader(flowvideo_dataset,  batch_size=8, shuffle=False)
    dualstream_loader = DataLoader(dualstream_dataset, batch_size=8, shuffle=False)

    # for frames, labels in frameimage_loader:
    #     print(frames.shape, labels.shape) # [batch, channels, height, width]

    # for video_frames, labels in framevideolist_loader:
    #     print(45*'-')
    #     for frame in video_frames: # loop through number of frames
    #         print(frame.shape, labels.shape)# [batch, channels, height, width]

    # for video_frames, labels in framevideostack_loader:
    #     print(video_frames.shape, labels.shape) # [batch, channels, number of frames, height, width]

    for spatial_input, temporal_input, labels in dualstream_loader:
        print(f"Spatial Input Shape: {spatial_input.shape}")  # [batch, 3, T, H, W]
        print(f"Temporal Input Shape: {temporal_input.shape}")  # [batch, 2*(T-1), H, W]
        print(f"Labels Shape: {labels.shape}")  # [batch]
            
