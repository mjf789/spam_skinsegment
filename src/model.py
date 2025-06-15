"""Model utilities for skin segmentation."""
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from typing import Dict, Optional


def create_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Create a DeepLabV3 model for skin segmentation.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        DeepLabV3 model
    """
    model = deeplabv3_resnet50(weights='DEFAULT' if pretrained else None)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Dict:
    """Load model checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load to
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    checkpoint_path: str,
    **kwargs
):
    """Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        scheduler: Scheduler state to save
        epoch: Current epoch
        checkpoint_path: Path to save checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        **kwargs
    }
    torch.save(checkpoint, checkpoint_path)
