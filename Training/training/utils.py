import torch
import csv
import time
from pathlib import Path


def find_latest_checkpoint(checkpoint_dir):
    """Find latest checkpoint to resume from"""
    checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
    if not checkpoints:
        return None, 0
    
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


def save_checkpoint(path, epoch, student, teacher, head_s, head_t, predictor,
                   optimizer, scheduler, scaler, loss, dino_loss):
    """Save training checkpoint"""
    def get_state(model):
        return model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    torch.save({
        'epoch': epoch,
        'student': get_state(student),
        'teacher': get_state(teacher),
        'head_student': get_state(head_s),
        'head_teacher': get_state(head_t),
        'predictor': get_state(predictor),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'loss': loss,
        'center': dino_loss.center,
    }, path)


@torch.no_grad()
def update_teacher(student, teacher, head_s, head_t, momentum):
    """Update teacher with EMA"""
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        if ps.requires_grad:
            pt.data = momentum * pt.data + (1 - momentum) * ps.data
    
    for ps, pt in zip(head_s.parameters(), head_t.parameters()):
        pt.data = momentum * pt.data + (1 - momentum) * ps.data


def cosine_scheduler(base_value, final_value, total_steps, warmup_steps=0):
    """Cosine schedule with warmup"""
    import numpy as np
    warmup_schedule = np.linspace(base_value, final_value, warmup_steps) if warmup_steps > 0 else np.array([])
    
    iters = np.arange(total_steps - warmup_steps)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    
    schedule = np.concatenate((warmup_schedule, schedule))
    return schedule


def init_metrics_csv(output_dir):
    """Initialize CSV files for logging"""
    batch_csv = output_dir / 'batch_metrics.csv'
    epoch_csv = output_dir / 'epoch_metrics.csv'
    
    if not batch_csv.exists():
        with open(batch_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'epoch', 'batch', 'loss_total', 
                           'loss_dino', 'loss_ibot', 'loss_koleo', 
                           'gpu_gb', 'lr', 'teacher_temp'])
    
    if not epoch_csv.exists():
        with open(epoch_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'avg_loss', 'lr', 'throughput', 'batch_time'])
    
    return batch_csv, epoch_csv


def log_batch(csv_path, epoch, batch, losses, gpu_gb, lr, temp):
    """Log batch metrics"""
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            time.strftime('%Y-%m-%d %H:%M:%S'), epoch, batch,
            f'{losses["total"]:.6f}', f'{losses["dino"]:.6f}',
            f'{losses.get("ibot", 0):.6f}', f'{losses.get("koleo", 0):.6f}',
            f'{gpu_gb:.2f}', f'{lr:.8f}', f'{temp:.4f}'
        ])


def log_epoch(csv_path, epoch, avg_loss, lr, throughput, batch_time):
    """Log epoch metrics"""
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f'{avg_loss:.6f}', f'{lr:.8f}',
                        f'{throughput:.1f}', f'{batch_time:.3f}'])


def get_gpu_memory_gb():
    """Get GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def print_system_info(batch_size, use_mixed_precision):
    """Print system configuration"""
    print("="*80)
    print("ExPLoRA Training - System Info")
    print("="*80)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Batch size: {batch_size}")
    print(f"Mixed precision: {use_mixed_precision}")
    print("="*80)

