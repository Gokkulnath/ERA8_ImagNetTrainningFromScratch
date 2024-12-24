import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ImageNetStreamingDataset(IterableDataset):
    def __init__(self, split="train", transform=None, max_samples=None):
        # Get token from environment variable
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables. Please set it in .env file")
            
        self.dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            split=split,
            streaming=True,
            trust_remote_code=True,
            token=hf_token
        )
        self.transform = transform
        self.max_samples = max_samples
        
    def __iter__(self):
        count = 0
        for item in self.dataset:
            if self.max_samples and count >= self.max_samples:
                break
                
            # Convert to PIL Image if it's not already
            image = item["image"]
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
                
            # Convert grayscale to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            label = item["label"]
            
            if self.transform:
                image = self.transform(image)
                
            yield image, label
            count += 1

def get_transforms(config, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]) 