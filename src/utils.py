"""Utility functions for skin segmentation."""
import os
import torch
from typing import Optional


def setup_device() -> torch.device:
    """Setup and return the best available device.
    
    Returns:
        torch.device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("Using CPU")
    
    return device


def get_batch_size(device: torch.device, default: int = 4) -> int:
    """Get appropriate batch size based on GPU.
    
    Args:
        device: Torch device
        default: Default batch size
        
    Returns:
        Recommended batch size
    """
    if device.type != "cuda":
        return default
    
    gpu_name = torch.cuda.get_device_name(0)
    
    if "A100" in gpu_name:
        batch_size = 12
        print(f"Detected A100 GPU - using batch size {batch_size}")
    elif "V100" in gpu_name or "L4" in gpu_name:
        batch_size = 6
        print(f"Detected {gpu_name} - using batch size {batch_size}")
    else:  # T4 or other
        batch_size = 4
        print(f"Detected {gpu_name} - using batch size {batch_size}")
    
    return batch_size


class Logger:
    """Simple logger that writes to both console and file."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log(self, message: str):
        """Log message to console and file."""
        print(message)
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(f"{message}\n")
            except Exception as e:
                print(f"Warning: Could not write to log file: {e}")
