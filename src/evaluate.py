"""Evaluation utilities for skin segmentation."""
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List


def eval_iou_per_image(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Evaluate mean IoU per image.
    
    Args:
        model: Model to evaluate
        loader: DataLoader for evaluation
        device: Device to run on
        
    Returns:
        Mean IoU across all images
    """
    per_image_ious = []
    model.eval()
    
    with torch.no_grad():
        for imgs, msks in loader:
            imgs, msks = imgs.to(device), msks.to(device)
            out = model(imgs)['out']
            preds = out.argmax(1)
            
            for pred, true in zip(preds, msks):
                pred_mask = (pred == 1)
                true_mask = (true == 1)
                inter = torch.logical_and(pred_mask, true_mask).sum().item()
                uni = torch.logical_or(pred_mask, true_mask).sum().item()
                per_image_ious.append(inter/uni if uni > 0 else float("nan"))
    
    return float(np.nanmean(per_image_ious))
