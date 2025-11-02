#!/usr/bin/env python3
"""
Utility functions for DinoV3 training: checkpoint management, logging, and model creation.
"""

import os
import csv
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import timm


# ============================================================================
# Checkpoint Management
# ============================================================================

def find_latest_checkpoint(checkpoint_dir: Path) -> Tuple[Optional[Path], int]:
    """
    Find the latest checkpoint to resume from.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        Tuple of (checkpoint_path, epoch_number). Returns (None, 0) if no checkpoint found.
    """
    checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
    if not checkpoints:
        return None, 0
    
    # Extract epoch numbers and find the latest
    epoch_nums = []
    for ckpt in checkpoints:
        try:
            epoch_num = int(ckpt.stem.split('_')[-1])
            epoch_nums.append((epoch_num, ckpt))
        except:
            continue
    
    if not epoch_nums:
        return None, 0
    
    latest_epoch, latest_ckpt = max(epoch_nums, key=lambda x: x[0])
    return latest_ckpt, latest_epoch


def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    student: nn.Module,
    teacher: nn.Module,
    proj_student: nn.Module,
    proj_teacher: nn.Module,
    predictor: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    loss: float
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        checkpoint_path: Path to save the checkpoint
        epoch: Current epoch number
        student: Student model
        teacher: Teacher model
        proj_student: Student projection head
        proj_teacher: Teacher projection head
        predictor: Predictor head (BYOL)
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
        loss: Current loss value
    """
    # Extract state_dict from DataParallel if needed
    student_state = student.module.state_dict() if hasattr(student, 'module') else student.state_dict()
    teacher_state = teacher.module.state_dict() if hasattr(teacher, 'module') else teacher.state_dict()
    proj_student_state = proj_student.module.state_dict() if hasattr(proj_student, 'module') else proj_student.state_dict()
    proj_teacher_state = proj_teacher.module.state_dict() if hasattr(proj_teacher, 'module') else proj_teacher.state_dict()
    predictor_state = predictor.module.state_dict() if hasattr(predictor, 'module') else predictor.state_dict()
    
    torch.save({
        'epoch': epoch,
        'student': student_state,
        'teacher': teacher_state,
        'proj_student': proj_student_state,
        'proj_teacher': proj_teacher_state,
        'predictor': predictor_state,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    print(f"✓ Saved checkpoint to {checkpoint_path}")
    print(f"  File size: {checkpoint_path.stat().st_size / 1024**3:.2f}GB")


def extract_backbone(checkpoint_path: Path, output_path: Path, model_key: str = 'student') -> None:
    """
    Extract just the backbone model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the full checkpoint
        output_path: Path to save the extracted backbone
        model_key: Which model to extract ('student' or 'teacher')
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if model_key not in ckpt:
        raise KeyError(f"Key '{model_key}' not found in checkpoint. Available keys: {list(ckpt.keys())}")
    
    backbone_state_dict = ckpt[model_key]
    
    torch.save(backbone_state_dict, output_path)
    print(f"✓ Extracted {model_key} backbone to {output_path}")
    print(f"  Original size: {checkpoint_path.stat().st_size / 1024**3:.2f}GB")
    print(f"  Backbone size: {output_path.stat().st_size / 1024**3:.2f}GB")


# ============================================================================
# Model Creation
# ============================================================================

def create_dinov3_models(
    img_size: int = 384,
    pretrained_path: Optional[str] = None,
    checkpoint_data: Optional[Dict] = None,
    device: str = 'cuda',
    proj_dim: int = 1024,
    pred_dim: int = 4096,
    dynamic_img_size: bool = True
) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Create student, teacher, projection heads, and predictor (BYOL-style).
    
    Args:
        img_size: Input image size (base size)
        pretrained_path: Path to pretrained weights (for fresh start)
        checkpoint_data: Loaded checkpoint dict (for resuming)
        device: Device to move models to
        proj_dim: Projection dimension (default 1024, no bottleneck!)
        pred_dim: Predictor hidden dimension (default 4096)
        dynamic_img_size: Allow variable input sizes (default True, needed for multi-crop)
        
    Returns:
        Tuple of (student, teacher, projection_student, projection_teacher, predictor)
    """
    # Create student with dynamic image size support
    student = timm.create_model(
        'vit_large_patch16_384', 
        pretrained=False, 
        num_classes=0, 
        img_size=img_size,
        dynamic_img_size=dynamic_img_size  # CRITICAL: Allows any input size!
    )
    
    if checkpoint_data is not None:
        # Resume from checkpoint
        student.load_state_dict(checkpoint_data['student'])
        print("✓ Loaded student from checkpoint")
    elif pretrained_path and Path(pretrained_path).exists():
        # Load pretrained weights
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        student.load_state_dict(checkpoint, strict=False)
        print(f"✓ Loaded pretrained weights from {pretrained_path}")
        del checkpoint
        torch.cuda.empty_cache()
    else:
        print("⚠ No pretrained weights loaded - training from scratch")
    
    # Create teacher with dynamic image size support
    teacher = timm.create_model(
        'vit_large_patch16_384', 
        pretrained=False, 
        num_classes=0, 
        img_size=img_size,
        dynamic_img_size=dynamic_img_size  # CRITICAL: Allows any input size!
    )
    
    if checkpoint_data is not None:
        teacher.load_state_dict(checkpoint_data['teacher'])
        print("✓ Loaded teacher from checkpoint")
    else:
        teacher.load_state_dict(student.state_dict())
        print("✓ Initialized teacher from student")
    
    # Move to device
    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    
    # Verify dynamic image size is enabled
    if dynamic_img_size:
        print(f"✓ Dynamic image size ENABLED - can handle variable resolutions")
        print(f"  Base size: {img_size}×{img_size}")
        print(f"  Can process any size (e.g., 96×96, 384×384, 518×518)")
    
    # Create projection heads - NO BOTTLENECK! Keep at proj_dim (1024)
    projection_student = nn.Sequential(
        nn.Linear(1024, proj_dim),
        nn.BatchNorm1d(proj_dim),
        nn.ReLU(inplace=True),
        nn.Linear(proj_dim, proj_dim)
    ).to(device)
    
    projection_teacher = nn.Sequential(
        nn.Linear(1024, proj_dim),
        nn.BatchNorm1d(proj_dim),
        nn.ReLU(inplace=True),
        nn.Linear(proj_dim, proj_dim)
    ).to(device)
    
    # Create predictor - BYOL-style (only on student side)
    predictor = nn.Sequential(
        nn.Linear(proj_dim, pred_dim),
        nn.BatchNorm1d(pred_dim),
        nn.ReLU(inplace=True),
        nn.Linear(pred_dim, proj_dim)
    ).to(device)
    
    if checkpoint_data is not None:
        projection_student.load_state_dict(checkpoint_data['proj_student'])
        projection_teacher.load_state_dict(checkpoint_data['proj_teacher'])
        if 'predictor' in checkpoint_data:
            predictor.load_state_dict(checkpoint_data['predictor'])
        print("✓ Loaded projection heads and predictor from checkpoint")
    else:
        projection_teacher.load_state_dict(projection_student.state_dict())
        print("✓ Initialized projection heads and predictor")
    
    projection_teacher.eval()
    for p in projection_teacher.parameters():
        p.requires_grad = False
    
    return student, teacher, projection_student, projection_teacher, predictor


# ============================================================================
# Logging
# ============================================================================

def init_metrics_files(output_dir: Path) -> Tuple[Path, Path]:
    """
    Initialize CSV metrics files if they don't exist.
    
    Args:
        output_dir: Directory to save metrics files
        
    Returns:
        Tuple of (realtime_metrics_path, epoch_metrics_path)
    """
    realtime_metrics_file = output_dir / 'realtime_metrics.csv'
    epoch_metrics_file = output_dir / 'epoch_metrics.csv'
    
    if not realtime_metrics_file.exists():
        with open(realtime_metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'epoch', 'batch', 'loss', 'student_teacher_similarity', 
                           'gpu_memory_gb', 'samples_per_sec', 'learning_rate', 'eta_minutes'])
    
    if not epoch_metrics_file.exists():
        with open(epoch_metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'avg_loss', 'learning_rate', 'avg_throughput', 'avg_batch_time'])
    
    return realtime_metrics_file, epoch_metrics_file


def log_batch_metrics(
    metrics_file: Path,
    epoch: int,
    batch_idx: int,
    loss: float,
    similarity: float,
    gpu_memory_gb: float,
    samples_per_sec: float,
    learning_rate: float,
    eta_minutes: float
) -> None:
    """Log metrics for a single batch."""
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            time.strftime('%Y-%m-%d %H:%M:%S'),
            epoch,
            batch_idx,
            f'{loss:.6f}',
            f'{similarity:.6f}',
            f'{gpu_memory_gb:.2f}',
            f'{samples_per_sec:.1f}',
            f'{learning_rate:.8f}',
            f'{eta_minutes:.1f}'
        ])


def log_epoch_metrics(
    metrics_file: Path,
    epoch: int,
    avg_loss: float,
    learning_rate: float,
    avg_throughput: float,
    avg_batch_time: float
) -> None:
    """Log metrics for a completed epoch."""
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f'{avg_loss:.6f}',
            f'{learning_rate:.8f}',
            f'{avg_throughput:.1f}',
            f'{avg_batch_time:.3f}'
        ])


# ============================================================================
# Training Utilities
# ============================================================================

def update_teacher(student, teacher, proj_student, proj_teacher, predictor=None, momentum=0.996):
    """Update teacher models using exponential moving average."""
    with torch.no_grad():
        for ps, pt in zip(student.parameters(), teacher.parameters()):
            pt.data = momentum * pt.data + (1 - momentum) * ps.data
        for ps, pt in zip(proj_student.parameters(), proj_teacher.parameters()):
            pt.data = momentum * pt.data + (1 - momentum) * ps.data
        # Note: predictor is NOT updated - it's student-only in BYOL


def cosine_loss(p, z):
    """BYOL-style cosine loss: 2 - 2*cosine_similarity (always positive, optimal at 0)"""
    p = nn.functional.normalize(p, dim=-1)
    z = nn.functional.normalize(z, dim=-1)
    return 2 - 2 * (p * z).sum(dim=-1).mean()


def contrastive_loss(p, z):
    """Alternative contrastive loss (kept for backwards compatibility)"""
    p = nn.functional.normalize(p, dim=-1)
    z = nn.functional.normalize(z, dim=-1)
    return -2 * (p * z).sum(dim=-1).mean()


def get_memory_usage():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def print_system_info(batch_size: int, use_mixed_precision: bool = True):
    """Print system and training configuration information."""
    print("="*80)
    print("DinoV3 Training - System Info")
    print("="*80)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Total GPU Memory per device: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print(f"Total Batch Size: {batch_size}")
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"  → {batch_size // torch.cuda.device_count()} samples per GPU × {torch.cuda.device_count()} GPUs")
    print(f"Mixed Precision (FP16): {use_mixed_precision}")
    print("="*80)
