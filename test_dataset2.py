import torch
from torchvision import transforms as T
from PIL import Image
import numpy as np
import os
from datasets2 import FrameVideoDataset, get_transforms
import matplotlib.pyplot as plt

def save_tensor_as_image(tensor, filepath):
    """Convert a tensor to a PIL Image and save it."""
    # Denormalize if needed (assuming data was normalized with ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    img = tensor * std + mean
    img = img.clamp(0, 1)
    
    # Convert to PIL Image and save
    img = T.ToPILImage()(img)
    img.save(filepath)

def plot_frames(frames, output_path):
    """Plot multiple frames in a grid with correct colors."""
    n_frames = len(frames)
    n_cols = 5
    n_rows = (n_frames + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for i in range(n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        
        if i < n_frames:
            # Denormalize and convert to numpy
            frame = frames[i] * std + mean
            frame = frame.clamp(0, 1)
            frame = frame.permute(1, 2, 0).numpy()
            
            axes[row, col].imshow(frame)
            axes[row, col].set_title(f'Frame {i}')
        
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

def test_transform_consistency():
    # Create output directory
    output_dir = "transform_consistency_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dataset with video consistency transforms
    data_transforms = get_transforms(rotation_degree=30, transform_size=224, video_consistency=True)
    dataset = FrameVideoDataset(
        root_dir='./data/ufc10',  # Update this path as needed
        split='train',
        transform=data_transforms['train'],
        stack_frames=False  # Keep as list to examine individual frames
    )
    
    # Test multiple videos
    for video_idx in range(min(5, len(dataset))):
        frames, label = dataset[video_idx]
        
        # Save individual frames
        for i, frame in enumerate(frames):
            save_tensor_as_image(
                frame,
                os.path.join(output_dir, f'video_{video_idx}_frame_{i}.jpg')
            )
        
        # Save composite visualization
        plot_frames(
            frames,
            os.path.join(output_dir, f'video_{video_idx}_all_frames.jpg')
        )
        
        print(f"Processed video {video_idx}")

def verify_transform_consistency():
    """
    Additional function to quantitatively verify transform consistency
    """
    data_transforms = get_transforms(rotation_degree=30, transform_size=224, video_consistency=True)
    dataset = FrameVideoDataset(
        root_dir='./data/ufc10',  # Update this path as needed
        split='train',
        transform=data_transforms['train'],
        stack_frames=False
    )
    
    results = []
    
    # Test multiple videos
    for video_idx in range(min(5, len(dataset))):
        frames, _ = dataset[video_idx]
        frames = torch.stack(frames)  # [num_frames, channels, height, width]
        
        # Calculate frame-to-frame differences
        frame_diffs = torch.abs(frames[1:] - frames[:-1]).mean()
        
        # Calculate standard deviation of pixel values across frames
        frame_std = torch.std(frames, dim=0).mean()
        
        results.append({
            'video_idx': video_idx,
            'frame_diffs': frame_diffs.item(),
            'frame_std': frame_std.item()
        })
        
        print(f"Video {video_idx}:")
        print(f"Average frame-to-frame difference: {frame_diffs:.4f}")
        print(f"Average pixel standard deviation across frames: {frame_std:.4f}")
        print("-" * 50)

if __name__ == '__main__':
    # Run both visual and quantitative tests
    test_transform_consistency()
    verify_transform_consistency()