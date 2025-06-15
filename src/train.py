"""Training utilities for skin segmentation."""
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from typing import Dict, Tuple, Optional

from .dataset import SkinDataset, create_transform
from .model import create_model, save_checkpoint, load_checkpoint
from .evaluate import eval_iou_per_image
from .utils import Logger, get_batch_size
from .config import Config


def create_data_loaders(config: Config, device: torch.device) -> Dict[str, DataLoader]:
    """Create all data loaders from configuration.
    
    Args:
        config: Configuration object
        device: Device to use
        
    Returns:
        Dictionary of data loaders
    """
    transform = create_transform(config.training.image_size)
    batch_size = get_batch_size(device, config.training.batch_size)
    
    # Create datasets
    fsd_train_ds = SkinDataset(
        config.data.fsd_train_images,
        config.data.fsd_train_masks,
        transform
    )
    hand_train_ds = SkinDataset(
        config.data.hand_train_images,
        config.data.hand_train_masks,
        transform
    )
    train_ds = ConcatDataset([fsd_train_ds, hand_train_ds])
    
    fsd_val_ds = SkinDataset(
        config.data.fsd_val_images,
        config.data.fsd_val_masks,
        transform
    )
    hand_val_ds = SkinDataset(
        config.data.hand_val_images,
        config.data.hand_val_masks,
        transform
    )
    
    # Weighted sampling
    fsd_weights = [1.0] * len(fsd_train_ds)
    hand_weights = [config.training.hand_weight] * len(hand_train_ds)
    weights = fsd_weights + hand_weights
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    # Create loaders
    loaders = {
        'train': DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=config.training.num_workers, pin_memory=True, drop_last=True
        ),
        'val_fsd': DataLoader(
            fsd_val_ds, batch_size=batch_size, shuffle=False,
            num_workers=config.training.num_workers, pin_memory=True
        ),
        'val_hand': DataLoader(
            hand_val_ds, batch_size=batch_size, shuffle=False,
            num_workers=config.training.num_workers, pin_memory=True
        )
    }
    
    # Optional hand-only validation
    if config.data.hand_val_hand_masks:
        hand_val_hand_ds = SkinDataset(
            config.data.hand_val_images,
            config.data.hand_val_hand_masks,
            transform
        )
        loaders['val_hand_only'] = DataLoader(
            hand_val_hand_ds, batch_size=batch_size, shuffle=False,
            num_workers=config.training.num_workers, pin_memory=True
        )
    
    return loaders


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler],
    epoch: int,
    logger: Logger,
    log_interval: int = 100
) -> float:
    """Train for one epoch.
    
    Args:
        model: Model to train
        loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        scaler: GradScaler for mixed precision
        epoch: Current epoch number
        logger: Logger instance
        log_interval: Batch logging interval
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for i, (imgs, msks) in enumerate(loader):
        imgs, msks = imgs.to(device), msks.to(device)
        optimizer.zero_grad()
        
        if scaler:
            with autocast(device_type='cuda'):
                out = model(imgs)['out']
                loss = criterion(out, msks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)['out']
            loss = criterion(out, msks)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * imgs.size(0)
        
        if (i + 1) % log_interval == 0:
            logger.log(f"  Epoch {epoch}, Batch {i+1}/{len(loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(loader.dataset)


class Trainer:
    """Main trainer class for skin segmentation."""
    
    def __init__(self, config: Config, device: torch.device):
        """Initialize trainer.
        
        Args:
            config: Configuration object
            device: Device to use
        """
        self.config = config
        self.device = device
        self.logger = Logger(os.path.join(config.output_dir, config.log_file))
        
        # Create directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Initialize model
        self.model = create_model(num_classes=2, pretrained=True).to(device)
        
        # Initialize training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=config.training.lr_factor,
            patience=config.training.patience,
            verbose=True
        )
        self.scaler = GradScaler() if config.training.use_amp and device.type == "cuda" else None
        
        # Create data loaders
        self.loaders = create_data_loaders(config, device)
        
        # Training state
        self.start_epoch = 1
        self.best_hand_iou = 0.0
        self.best_fsd_iou = 0.0
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        if os.path.exists(checkpoint_path):
            self.logger.log(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = load_checkpoint(
                self.model, checkpoint_path, self.device,
                self.optimizer, self.scheduler
            )
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_hand_iou = checkpoint.get('best_hand_iou', 0.0)
            self.best_fsd_iou = checkpoint.get('best_fsd_iou', 0.0)
            self.logger.log(f"Resuming from epoch {self.start_epoch}")
    
    def train(self):
        """Run the full training loop."""
        self.logger.log(f"{'='*50}")
        self.logger.log(f"Training started on {self.device}")
        self.logger.log(f"Starting from epoch {self.start_epoch}")
        
        for epoch in range(self.start_epoch, self.config.training.num_epochs + 1):
            # Training
            epoch_start = time.time()
            avg_loss = train_epoch(
                self.model, self.loaders['train'], self.optimizer,
                self.criterion, self.device, self.scaler, epoch,
                self.logger, self.config.training.log_interval
            )
            
            # Evaluation
            fsd_iou = eval_iou_per_image(self.model, self.loaders['val_fsd'], self.device)
            hand_iou = eval_iou_per_image(self.model, self.loaders['val_hand'], self.device)
            
            # Update learning rate
            self.scheduler.step(hand_iou)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log results
            epoch_time = (time.time() - epoch_start) / 60
            self.logger.log(
                f"Epoch {epoch}/{self.config.training.num_epochs}: "
                f"Loss {avg_loss:.4f}, FSD mIoU {fsd_iou:.4f}, "
                f"Hand mIoU {hand_iou:.4f}, LR: {current_lr:.2e}, "
                f"Time: {epoch_time:.1f} min"
            )
            
            # Save best model
            if hand_iou > self.best_hand_iou:
                self.best_hand_iou = hand_iou
                save_path = os.path.join(self.config.checkpoint_dir, "best_model.pth")
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler, epoch, save_path,
                    best_hand_iou=self.best_hand_iou, fsd_iou=fsd_iou
                )
                self.logger.log(f" → New best Hand IoU: {hand_iou:.4f}")
            
            if fsd_iou > self.best_fsd_iou:
                self.best_fsd_iou = fsd_iou
                self.logger.log(f" → New best FSD IoU: {fsd_iou:.4f}")
            
            # Save checkpoints
            if epoch % self.config.training.checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    self.config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"
                )
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler, epoch, checkpoint_path,
                    best_hand_iou=self.best_hand_iou, best_fsd_iou=self.best_fsd_iou
                )
            
            # Always save latest
            latest_path = os.path.join(self.config.checkpoint_dir, "latest_checkpoint.pth")
            save_checkpoint(
                self.model, self.optimizer, self.scheduler, epoch, latest_path,
                best_hand_iou=self.best_hand_iou, best_fsd_iou=self.best_fsd_iou
            )
        
        self.logger.log(f"\nTraining completed!")
        self.logger.log(f"Best Hand IoU: {self.best_hand_iou:.4f}")
        self.logger.log(f"Best FSD IoU: {self.best_fsd_iou:.4f}")
