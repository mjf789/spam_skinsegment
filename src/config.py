"""Configuration management for skin segmentation project."""
import os
from dataclasses import dataclass
from typing import Optional, Tuple
import yaml


@dataclass
class DataConfig:
    """Data-related configuration."""
    fsd_train_images: str
    fsd_train_masks: str
    fsd_val_images: str
    fsd_val_masks: str
    hand_train_images: str
    hand_train_masks: str
    hand_val_images: str
    hand_val_masks: str
    hand_val_hand_masks: Optional[str] = None
    
    def update_paths(self, base_path: str):
        """Update all paths to be relative to base_path."""
        for field in self.__dataclass_fields__:
            current_value = getattr(self, field)
            if current_value and not os.path.isabs(current_value):
                setattr(self, field, os.path.join(base_path, current_value))


@dataclass
class TrainingConfig:
    """Training-related configuration."""
    batch_size: int = 4
    num_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    image_size: Tuple[int, int] = (1024, 1024)
    num_workers: int = 2
    use_amp: bool = True
    hand_weight: float = 5.0
    checkpoint_interval: int = 5
    log_interval: int = 100
    patience: int = 2
    lr_factor: float = 0.5


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig
    training: TrainingConfig
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_file: str = "training_log.txt"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        data_config = DataConfig(**config_dict['data'])
        training_config = TrainingConfig(**config_dict['training'])
        
        return cls(
            data=data_config,
            training=training_config,
            output_dir=config_dict.get('output_dir', './outputs'),
            checkpoint_dir=config_dict.get('checkpoint_dir', './checkpoints'),
            log_file=config_dict.get('log_file', 'training_log.txt')
        )
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        config_dict = {
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'log_file': self.log_file
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


def get_colab_config() -> Config:
    """Get default configuration for Google Colab."""
    data_config = DataConfig(
        fsd_train_images="/content/drive/MyDrive/Brand Analysis Study/Deeplabv3/FSD with Random Crop/train/images",
        fsd_train_masks="/content/drive/MyDrive/Brand Analysis Study/Deeplabv3/FSD with Random Crop/train/masks",
        fsd_val_images="/content/drive/MyDrive/Brand Analysis Study/Deeplabv3/FSD with Random Crop/val/images",
        fsd_val_masks="/content/drive/MyDrive/Brand Analysis Study/Deeplabv3/FSD with Random Crop/val/masks",
        hand_train_images="/content/drive/MyDrive/Brand Analysis Study/Deeplabv3/Social Media Tests/V2: Hand Training/hand_train/hand_train_pics",
        hand_train_masks="/content/drive/MyDrive/Brand Analysis Study/Deeplabv3/Social Media Tests/V2: Hand Training/hand_train/hand_train_masks",
        hand_val_images="/content/drive/MyDrive/Brand Analysis Study/Deeplabv3/Social Media Tests/V2: Hand Training/hand_val/hand_val_pics",
        hand_val_masks="/content/drive/MyDrive/Brand Analysis Study/Deeplabv3/Social Media Tests/V2: Hand Training/hand_val/hand_val_masks",
        hand_val_hand_masks="/content/drive/MyDrive/Brand Analysis Study/Deeplabv3/Social Media Tests/V2: Hand Training/hand_val_hand"
    )
    
    training_config = TrainingConfig()
    
    return Config(
        data=data_config,
        training=training_config,
        output_dir="/content/drive/MyDrive/Brand Analysis Study/Models",
        checkpoint_dir="/content/drive/MyDrive/Brand Analysis Study/Models",
        log_file="training_log.txt"
    )
