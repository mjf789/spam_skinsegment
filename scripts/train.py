#!/usr/bin/env python
"""Command-line training script."""
import argparse
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.utils import setup_device
from src.train import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train skin segmentation model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None,
        help="Override number of epochs"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_yaml(args.config)
    
    # Override epochs if specified
    if args.epochs:
        config.training.num_epochs = args.epochs
    
    # Setup device
    device = setup_device()
    
    # Initialize trainer
    trainer = Trainer(config, device)
    
    # Resume if specified
    if args.resume:
        trainer.resume_from_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
