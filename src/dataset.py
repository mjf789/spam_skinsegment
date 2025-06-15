"""Dataset classes for skin segmentation."""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from typing import Callable, Optional, Tuple


class SkinDataset(Dataset):
    """Dataset for skin segmentation images and masks."""
    
    def __init__(
        self, 
        images_dir: str, 
        masks_dir: str, 
        transform: Optional[Callable] = None
    ):
        """Initialize the dataset.
        
        Args:
            images_dir: Directory containing input images
            masks_dir: Directory containing segmentation masks
            transform: Optional transform function
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.files = sorted(
            f for f in os.listdir(images_dir)
            if os.path.isfile(os.path.join(images_dir, f))
        )
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (image_tensor, mask_tensor)
        """
        img_name = self.files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            image, mask = self.transform(image, mask)
        
        image_tensor = F.to_tensor(image)
        mask_arr = (np.array(mask) > 127).astype(np.int64)
        mask_tensor = torch.from_numpy(mask_arr)
        
        return image_tensor, mask_tensor


def create_transform(image_size: Tuple[int, int]) -> Callable:
    """Create a basic transform for images and masks.
    
    Args:
        image_size: Target size (height, width)
        
    Returns:
        Transform function
    """
    def transform(image, mask):
        image = F.resize(image, image_size, interpolation=InterpolationMode.BILINEAR)
        mask = F.resize(mask, image_size, interpolation=InterpolationMode.NEAREST)
        return image, mask
    
    return transform
