import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import time
import json
import sys
from torch.cuda.amp import autocast, GradScaler

from dinoTools import (
    find_latest_checkpoint,
    create_dinov3_models,
    init_metrics_files,
    log_batch_metrics,
    log_epoch_metrics,
    update_teacher,
    get_memory_usage,
    print_system_info
)


# ============================================================================
# Load Configuration
# ============================================================================
def load_config(config_path='config.json'):
    """Load configuration from JSON file"""
    if not Path(config_path).exists():
        print(f"ERROR: Config file not found: {config_path}")
        print(f"Please copy config.json.example to config.json and update with your paths")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


# Load configuration
print("Loading configuration from config.json...")
config = load_config()

# Extract configuration values
DATA_DIRS = config['data_dirs']
PRETRAINED = config['pretrained_model']
OUTPUT_DIR = Path(config['output_dir'])
CHECKPOINT_DIR = Path(config['checkpoint_dir'])

# Training hyperparameters
training_config = config['training']
BATCH_SIZE = training_config['batch_size']
GLOBAL_CROP_SIZE = training_config['global_crop_size']
LOCAL_CROP_SIZE = training_config['local_crop_size']
N_LOCAL_CROPS = training_config['n_local_crops']
USE_MIXED_PRECISION = training_config['use_mixed_precision']
TOTAL_EPOCHS = training_config['total_epochs']
LR = training_config['learning_rate']
WEIGHT_DECAY = training_config['weight_decay']
EMA_MOMENTUM = training_config['ema_momentum']
CENTER_MOMENTUM = training_config['center_momentum']
TEACHER_TEMP = training_config['teacher_temp']
TEACHER_TEMP_WARMUP_EPOCHS = training_config['teacher_temp_warmup_epochs']
STUDENT_TEMP = training_config['student_temp']

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda')


# Print System Info
print_system_info(BATCH_SIZE, USE_MIXED_PRECISION)
print(f"\nMULTI-CROP SETUP:")
print(f"  Global crops: 2 @ {GLOBAL_CROP_SIZE}x{GLOBAL_CROP_SIZE}")
print(f"  Local crops: {N_LOCAL_CROPS} @ {LOCAL_CROP_SIZE}x{LOCAL_CROP_SIZE}")
print(f"  Total views per image: {2 + N_LOCAL_CROPS}")
print(f"  Teacher sees: 2 global views only")
print(f"  Student sees: ALL {2 + N_LOCAL_CROPS} views")
print(f"  Temperature sharpening: {TEACHER_TEMP} -> 0.07")
print(f"  Resolution: {GLOBAL_CROP_SIZE}px")

# Multi-Crop Dataset with Centering
class MultiCropAugmentation:
    """Creates 2 global + N local crops like DINOv3"""
    
    def __init__(self, global_size=518, local_size=96, n_local=8):
        # Global crops - large views (40-100% of image)
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Local crops - small patches (5-40% of image) 
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=(0.05, 0.4)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.n_local = n_local
    
    def __call__(self, img):
        """Returns [global1, global2, local1, ..., local8]"""
        crops = []
        # 2 global crops
        crops.append(self.global_transform(img))
        crops.append(self.global_transform(img))
        # N local crops
        for _ in range(self.n_local):
            crops.append(self.local_transform(img))
        return crops


class MultiCropDataset(Dataset):
    """Fast dataset with multi-crop augmentation"""
    
    def __init__(self, roots, transform=None):
        self.samples = []
        
        print(f"Loading dataset...")
        for root in roots:
            root_path = Path(root)
            if not root_path.exists():
                print(f"  Warning: {root} does not exist, skipping")
                continue
            
            try:
                folder = ImageFolder(str(root_path))
                self.samples.extend(folder.samples)
                print(f"  Loaded {len(folder.samples):,} images from {root}")
            except Exception as e:
                print(f"  Warning: Failed to load {root}: {e}")
        
        if len(self.samples) == 0:
            raise ValueError("No images found!")
        
        self.transform = transform
        print(f"Total: {len(self.samples):,} images\n")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, _ = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                crops = self.transform(img)
            return crops
        except Exception as e:
            print(f"  Warning: Error loading {path}: {e}")
            return self.__getitem__((idx + 1) % len(self))


# DINOv3 Loss with Centering
class DINOLossWithCentering(nn.Module):
    """
    DINOv3 loss with:
    - Centering (running mean subtraction)
    - Temperature sharpening
    - Cross-entropy between student and teacher
    """
    
    def __init__(self, out_dim, teacher_temp, student_temp, center_momentum):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        
        print(f"\nDINOLoss with Centering:")
        print(f"  Teacher temp: {teacher_temp}")
        print(f"  Student temp: {student_temp}")
        print(f"  Center momentum: {center_momentum}")
    
    def forward(self, student_output, teacher_output):
        """
        Args:
            student_output: list of 10 tensors [B, D] (2 global + 8 local)
            teacher_output: list of 2 tensors [B, D] (only global)
        Returns:
            loss: scalar
            teacher_probs: for monitoring
        """
        # Apply centering and temperature to teacher
        teacher_out_centered = [(t - self.center) / self.teacher_temp 
                                for t in teacher_output]
        teacher_probs = [F.softmax(t, dim=-1) for t in teacher_out_centered]
        
        # Apply temperature to student (no centering!)
        student_logits = [s / self.student_temp for s in student_output]
        student_log_probs = [F.log_softmax(s, dim=-1) for s in student_logits]
        
        # Cross-entropy loss between all student crops and teacher global crops
        total_loss = 0
        n_loss_terms = 0
        
        for t_idx, t_prob in enumerate(teacher_probs):
            for s_idx, s_log_prob in enumerate(student_log_probs):
                # Skip same global-to-global pairing
                if s_idx < 2 and t_idx == s_idx:
                    continue
                
                # Cross-entropy: -sum(teacher_prob * log(student_prob))
                loss = torch.sum(-t_prob * s_log_prob, dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        
        total_loss /= n_loss_terms
        
        # Update center with EMA
        self.update_center(teacher_output)
        
        return total_loss, teacher_probs[0]
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """EMA update of center"""
        batch_center = torch.cat(teacher_output).mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + \
                     batch_center * (1 - self.center_momentum)


# Custom Checkpoint Saving (includes center)
def save_checkpoint_with_center(checkpoint_path, epoch, student, teacher, proj_student, 
                               proj_teacher, predictor, optimizer, scheduler, scaler, 
                               loss, dino_loss):
    """Save checkpoint including DINOLoss center"""
    student_state = student.module.state_dict() if hasattr(student, 'module') else student.state_dict()
    teacher_state = teacher.module.state_dict() if hasattr(teacher, 'module') else teacher.state_dict()
    proj_s_state = proj_student.module.state_dict() if hasattr(proj_student, 'module') else proj_student.state_dict()
    proj_t_state = proj_teacher.module.state_dict() if hasattr(proj_teacher, 'module') else proj_teacher.state_dict()
    pred_state = predictor.module.state_dict() if hasattr(predictor, 'module') else predictor.state_dict()
    
    torch.save({
        'epoch': epoch,
        'student': student_state,
        'teacher': teacher_state,
        'proj_student': proj_s_state,
        'proj_teacher': proj_t_state,
        'predictor': pred_state,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'loss': loss,
        'center': dino_loss.center,  # Save center!
    }, checkpoint_path)



print("CREATING DATASET")

multi_crop_transform = MultiCropAugmentation(
    global_size=GLOBAL_CROP_SIZE,
    local_size=LOCAL_CROP_SIZE,
    n_local=N_LOCAL_CROPS
)

dataset = MultiCropDataset(DATA_DIRS, transform=multi_crop_transform)
dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=12, 
    pin_memory=True, 
    drop_last=True
)

# Model Creation
print("CREATING MODELS")

latest_checkpoint, start_epoch = find_latest_checkpoint(CHECKPOINT_DIR)

if latest_checkpoint and start_epoch >= TOTAL_EPOCHS:
    print(f"\nTraining complete! Epoch {start_epoch}/{TOTAL_EPOCHS}")
    exit(0)

current_epoch = start_epoch + 1

if latest_checkpoint:
    print(f"\nResuming from: {latest_checkpoint}")
    print(f"   Epoch {current_epoch}/{TOTAL_EPOCHS}")
    checkpoint_data = torch.load(latest_checkpoint, map_location='cpu')
else:
    print(f"\nStarting fresh training")
    print(f"   Epoch {current_epoch}/{TOTAL_EPOCHS}")
    checkpoint_data = None

student, teacher, projection_student, projection_teacher, predictor = create_dinov3_models(
    img_size=GLOBAL_CROP_SIZE,  # 518px base size
    pretrained_path=PRETRAINED if not checkpoint_data else None,
    checkpoint_data=checkpoint_data,
    device=device,
    proj_dim=1024,  # No bottleneck!
    pred_dim=4096,  # BYOL predictor
    dynamic_img_size=True  # CRITICAL: Allows 96px local crops AND 518px global crops!
)

# Multi-GPU
if torch.cuda.device_count() > 1:
    print(f"\nDataParallel across {torch.cuda.device_count()} GPUs")
    student = nn.DataParallel(student)
    teacher = nn.DataParallel(teacher)
    projection_student = nn.DataParallel(projection_student)
    projection_teacher = nn.DataParallel(projection_teacher)
    predictor = nn.DataParallel(predictor)

print(f"\nModels ready!")

dino_loss = DINOLossWithCentering(
    out_dim=1024,
    teacher_temp=TEACHER_TEMP,
    student_temp=STUDENT_TEMP,
    center_momentum=CENTER_MOMENTUM
).to(device)

# Restore center if resuming
if checkpoint_data and 'center' in checkpoint_data:
    dino_loss.center = checkpoint_data['center'].to(device)
    print(f"Restored centering buffer")

optimizer = optim.AdamW([
    {'params': student.parameters()},
    {'params': projection_student.parameters()},
    {'params': predictor.parameters()}
], lr=LR, weight_decay=WEIGHT_DECAY)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)
scaler = GradScaler(enabled=USE_MIXED_PRECISION)

if checkpoint_data:
    optimizer.load_state_dict(checkpoint_data['optimizer'])
    scheduler.load_state_dict(checkpoint_data['scheduler'])
    if 'scaler' in checkpoint_data:
        scaler.load_state_dict(checkpoint_data['scaler'])
    print(f"Optimizer restored | LR: {scheduler.get_last_lr()[0]:.2e}")
    del checkpoint_data

# ============================================================================
# Metrics Logging
# ============================================================================
realtime_metrics_file, epoch_metrics_file = init_metrics_files(OUTPUT_DIR)

# ============================================================================
# Training Loop
# ============================================================================
print("\n" + "="*80)
print(f"STARTING TRAINING: EPOCHS {current_epoch}-{TOTAL_EPOCHS}")
print("="*80)

for epoch in range(current_epoch, TOTAL_EPOCHS + 1):
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch}/{TOTAL_EPOCHS}")
    print(f"{'='*80}")
    
    # Temperature warmup
    if epoch <= TEACHER_TEMP_WARMUP_EPOCHS:
        progress = epoch / TEACHER_TEMP_WARMUP_EPOCHS
        current_temp = TEACHER_TEMP + progress * (0.07 - TEACHER_TEMP)
        dino_loss.teacher_temp = current_temp
        print(f"Teacher temperature: {current_temp:.4f}")
    
    student.train()
    projection_student.train()
    predictor.train()
    
    epoch_loss = 0
    last_log_time = time.time()
    batch_times = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{TOTAL_EPOCHS}")
    for batch_idx, crops in enumerate(pbar):
        batch_start = time.time()
        
        # crops is a list of 10 tensors: [global1, global2, local1, ..., local8]
        # Move all to GPU
        crops = [c.to(device, non_blocking=True) for c in crops]
        global_crops = crops[:2]  # Teacher only sees these
        all_crops = crops  # Student sees all
        
        with autocast(enabled=USE_MIXED_PRECISION):
            # Student forward on ALL crops
            student_outputs = []
            for crop in all_crops:
                h = student(crop)
                z = projection_student(h)
                p = predictor(z)  # Apply predictor
                student_outputs.append(p)
            
            # Teacher forward on GLOBAL crops only (no predictor!)
            with torch.no_grad():
                teacher_outputs = []
                for crop in global_crops:
                    h = teacher(crop)
                    z = projection_teacher(h)
                    teacher_outputs.append(z)
            
            # DINOv3 loss with centering
            loss, teacher_probs = dino_loss(student_outputs, teacher_outputs)
            
            # Monitor entropy (higher = more diverse)
            with torch.no_grad():
                entropy = -(teacher_probs * torch.log(teacher_probs + 1e-8)).sum(dim=-1).mean()
        
        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update teacher with EMA
        update_teacher(student, teacher, projection_student, projection_teacher, 
                      predictor, momentum=EMA_MOMENTUM)
        
        epoch_loss += loss.item()
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        # Metrics
        samples_per_sec = BATCH_SIZE / batch_time
        avg_batch_time = sum(batch_times) / len(batch_times)
        eta_minutes = (avg_batch_time * (len(dataloader) - batch_idx - 1)) / 60.0
        
        # Log every 2 seconds
        if time.time() - last_log_time >= 2.0:
            log_batch_metrics(
                realtime_metrics_file, epoch, batch_idx + 1,
                loss.item(), entropy.item(), get_memory_usage(),
                samples_per_sec, scheduler.get_last_lr()[0], eta_minutes
            )
            last_log_time = time.time()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'entropy': f'{entropy.item():.2f}',
            'temp': f'{dino_loss.teacher_temp:.3f}',
            'gpu': f'{get_memory_usage():.1f}GB',
            'samp/s': f'{samples_per_sec:.0f}'
        })
    
    scheduler.step()
    
    # Epoch summary
    avg_loss = epoch_loss / len(dataloader)
    avg_batch_time = sum(batch_times) / len(batch_times)
    throughput = BATCH_SIZE / avg_batch_time
    
    log_epoch_metrics(epoch_metrics_file, epoch, avg_loss, 
                     scheduler.get_last_lr()[0], throughput, avg_batch_time)
    
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch} COMPLETE")
    print(f"{'='*80}")
    print(f"  Avg Loss: {avg_loss:.6f}")
    print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
    print(f"  Throughput: {throughput:.1f} samples/sec")
    print(f"  GPU Memory: {get_memory_usage():.1f}GB")
    print(f"{'='*80}\n")
    
    # Save checkpoint
    checkpoint_path = CHECKPOINT_DIR / f'checkpoint_epoch_{epoch}.pth'
    save_checkpoint_with_center(
        checkpoint_path, epoch, student, teacher,
        projection_student, projection_teacher, predictor,
        optimizer, scheduler, scaler, avg_loss, dino_loss
    )
    print(f"Saved: {checkpoint_path}\n")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"Trained epochs {current_epoch} to {TOTAL_EPOCHS}")
print(f"Checkpoints in: {CHECKPOINT_DIR}")
print(f"Multi-crop: 2 global @ {GLOBAL_CROP_SIZE}px + {N_LOCAL_CROPS} local @ {LOCAL_CROP_SIZE}px")
print(f"Centering: ENABLED")
print(f"Ready for VLA / segmentation tasks!")
print("="*80)
