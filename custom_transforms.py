# custom_transforms.py
import random
import numpy as np
import torch
from torchvision import transforms as T
from PIL import Image

class ConsistentRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, frames):
        if random.random() < self.p:
            return [T.functional.hflip(frame) for frame in frames]
        return frames

class ConsistentRandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees
        
    def __call__(self, frames):
        angle = T.RandomRotation.get_params([-self.degrees, self.degrees])
        return [T.functional.rotate(frame, angle) for frame in frames]

class ConsistentRandomResizedCrop:
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)
        
    def __call__(self, frames):
        # Get parameters for the first frame
        width, height = frames[0].size
        i, j, h, w = T.RandomResizedCrop.get_params(
            frames[0], 
            scale=(0.08, 1.0), 
            ratio=(3./4., 4./3.)
        )
        
        return [T.functional.resized_crop(
            frame, i, j, h, w, self.size, Image.BILINEAR
        ) for frame in frames]

class ConsistentColorJitter:
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        
    def __call__(self, frames):
        # Get random parameters once
        b = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        c = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        s = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        h = random.uniform(-self.hue, self.hue)
        
        return [T.functional.adjust_brightness(
            T.functional.adjust_contrast(
                T.functional.adjust_saturation(
                    T.functional.adjust_hue(frame, h),
                s),
            c),
        b) for frame in frames]