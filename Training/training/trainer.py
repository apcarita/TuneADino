import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import numpy as np

from models import create_explora_models
from losses import DINOLoss, iBOTLoss, KoleoLoss, random_masking
from data import MultiCropAugmentation, MultiCropDataset
from .utils import (
    find_latest_checkpoint, save_checkpoint, update_teacher,
    cosine_scheduler, init_metrics_csv, log_batch, log_epoch,
    get_gpu_memory_gb, print_system_info
)


class ExPLoRATrainer:
    """ExPLoRA (DINO + iBOT + LoRA) Trainer"""
    
    def __init__(self, config, data_dirs, pretrained_path, output_dir, checkpoint_dir, device='cuda'):
        self.cfg = config
        self.data_dirs = data_dirs
        self.pretrained_path = pretrained_path
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        
        # Print info
        print_system_info(config.batch_size, config.use_mixed_precision)
        print(f"\nExPLoRA Configuration:")
        print(f"  Resolution: {config.global_crop_size}x{config.global_crop_size} (global), {config.local_crop_size}x{config.local_crop_size} (local x{config.n_local_crops})")
        print(f"  Patches: {(config.global_crop_size // 16) ** 2} (sequence length)")
        print(f"  Learning rate: {config.learning_rate:.2e} (weight decay={config.weight_decay})")
        print(f"  LoRA: rank={config.lora_rank}, alpha={config.lora_alpha}")
        print(f"  Unfrozen blocks: {config.unfreeze_last_n_blocks}")
        print(f"  Gradient checkpointing: {config.gradient_checkpointing}")
        print(f"  Koleo weight: {config.koleo_weight}")
        print(f"  iBOT weight: {config.ibot_weight}")
        print(f"  Head frozen: first {config.head_frozen_iters} iters\n")
        
        # Setup
        self._setup_data()
        self._setup_models()
        self._setup_losses()
        self._setup_optimizer()
        self._setup_schedules()
        self._setup_metrics()
        
        print("\nSetup complete!\n")
    
    def _setup_data(self):
        """Setup dataset and dataloader"""
        print("Setting up data...")
        transform = MultiCropAugmentation(
            global_size=self.cfg.global_crop_size,
            local_size=self.cfg.local_crop_size,
            n_local=self.cfg.n_local_crops
        )
        
        dataset = MultiCropDataset(self.data_dirs, transform=transform)
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            drop_last=True
        )
        self.steps_per_epoch = len(self.dataloader)
        self.total_steps = self.steps_per_epoch * self.cfg.total_epochs
    
    def _setup_models(self):
        """Setup student, teacher, heads"""
        print("\nSetting up models...")
        
        # Check for checkpoint
        latest_ckpt, self.start_epoch = find_latest_checkpoint(self.checkpoint_dir)
        
        if latest_ckpt and self.start_epoch >= self.cfg.total_epochs:
            print(f"Training already complete at epoch {self.start_epoch}")
            exit(0)
        
        checkpoint_data = None
        if latest_ckpt:
            print(f"Resuming from: {latest_ckpt} (epoch {self.start_epoch})")
            checkpoint_data = torch.load(latest_ckpt, map_location='cpu')
        else:
            print("Starting fresh training")
        
        # Create models
        self.student, self.teacher, self.head_s, self.head_t, self.predictor, self.lora_params = \
            create_explora_models(
                img_size=self.cfg.global_crop_size,
                pretrained_path=self.pretrained_path if not checkpoint_data else None,
                checkpoint_data=checkpoint_data,
                device=self.device,
                lora_rank=self.cfg.lora_rank,
                lora_alpha=self.cfg.lora_alpha,
                unfreeze_last_n_blocks=self.cfg.unfreeze_last_n_blocks,
                drop_path_rate=self.cfg.drop_path_rate,
                gradient_checkpointing=self.cfg.gradient_checkpointing
            )
        
        # Multi-GPU
        if torch.cuda.device_count() > 1:
            print(f"\nUsing {torch.cuda.device_count()} GPUs")
            self.student = nn.DataParallel(self.student)
            self.teacher = nn.DataParallel(self.teacher)
            self.head_s = nn.DataParallel(self.head_s)
            self.head_t = nn.DataParallel(self.head_t)
            self.predictor = nn.DataParallel(self.predictor)
        
        self.checkpoint_data = checkpoint_data
    
    def _setup_losses(self):
        """Setup loss functions"""
        self.dino_loss = DINOLoss(
            out_dim=65536,
            teacher_temp=self.cfg.teacher_temp,
            student_temp=self.cfg.student_temp,
            center_momentum=0.9
        ).to(self.device)
        
        self.ibot_loss = iBOTLoss(
            out_dim=65536,
            teacher_temp=self.cfg.teacher_temp,
            student_temp=self.cfg.student_temp
        ).to(self.device)
        
        self.koleo_loss = KoleoLoss()
        
        if self.checkpoint_data and 'center' in self.checkpoint_data:
            self.dino_loss.center = self.checkpoint_data['center'].to(self.device)
            print("Restored DINO center")
    
    def _setup_optimizer(self):
        """Setup optimizer (no weight decay per paper)"""
        trainable_params = [
            {'params': [p for p in self.student.parameters() if p.requires_grad]},
            {'params': self.head_s.parameters()},
            {'params': self.predictor.parameters()}
        ]
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.total_epochs
        )
        
        self.scaler = GradScaler('cuda', enabled=self.cfg.use_mixed_precision)
        
        if self.checkpoint_data:
            self.optimizer.load_state_dict(self.checkpoint_data['optimizer'])
            self.scheduler.load_state_dict(self.checkpoint_data['scheduler'])
            if 'scaler' in self.checkpoint_data:
                self.scaler.load_state_dict(self.checkpoint_data['scaler'])
            print(f"Restored optimizer (LR: {self.scheduler.get_last_lr()[0]:.2e})")
    
    def _setup_schedules(self):
        """Setup temperature and momentum schedules"""
        # Teacher temperature: warmup from teacher_temp to teacher_temp_final
        self.temp_schedule = cosine_scheduler(
            self.cfg.teacher_temp,
            self.cfg.teacher_temp_final,
            self.total_steps,
            warmup_steps=self.cfg.teacher_temp_warmup_epochs * self.steps_per_epoch
        )
        
        # EMA momentum: warmup from initial to final
        self.ema_schedule = cosine_scheduler(
            self.cfg.ema_momentum_initial,
            self.cfg.ema_momentum_final,
            self.total_steps
        )
    
    def _setup_metrics(self):
        """Setup metric logging"""
        self.batch_csv, self.epoch_csv = init_metrics_csv(self.output_dir)
    
    def train(self):
        """Main training loop"""
        print("="*80)
        print(f"STARTING TRAINING: EPOCHS {self.start_epoch+1}-{self.cfg.total_epochs}")
        print("="*80)
        
        for epoch in range(self.start_epoch + 1, self.cfg.total_epochs + 1):
            self._train_epoch(epoch)
            
            # Save checkpoint
            avg_loss = self.epoch_loss / self.steps_per_epoch
            ckpt_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            save_checkpoint(
                ckpt_path, epoch, self.student, self.teacher,
                self.head_s, self.head_t, self.predictor,
                self.optimizer, self.scheduler, self.scaler,
                avg_loss, self.dino_loss
            )
            print(f"Saved checkpoint: {ckpt_path}\n")
        
        print("="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
    
    def _train_epoch(self, epoch):
        """Train one epoch"""
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{self.cfg.total_epochs}")
        print(f"{'='*80}")
        
        self.student.train()
        self.head_s.train()
        self.predictor.train()
        
        self.epoch_loss = 0
        batch_times = []
        last_log_time = time.time()
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        for batch_idx, crops in enumerate(pbar):
            batch_start = time.time()
            global_step = (epoch - 1) * self.steps_per_epoch + batch_idx
            
            # Update schedules
            self.dino_loss.teacher_temp = self.temp_schedule[global_step]
            self.ibot_loss.teacher_temp = self.temp_schedule[global_step]
            ema_momentum = self.ema_schedule[global_step]
            
            # Forward pass
            losses = self._train_step(crops, global_step)
            
            # Update teacher
            update_teacher(self.student, self.teacher, self.head_s, 
                         self.head_t, ema_momentum)
            
            # Metrics
            self.epoch_loss += losses['total']
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Logging
            if time.time() - last_log_time >= 2.0:
                log_batch(
                    self.batch_csv, epoch, batch_idx + 1, losses,
                    get_gpu_memory_gb(), self.scheduler.get_last_lr()[0],
                    self.dino_loss.teacher_temp
                )
                last_log_time = time.time()
            
            pbar.set_postfix({
                'loss': f'{losses["total"]:.4f}',
                'dino': f'{losses["dino"]:.4f}',
                'ibot': f'{losses.get("ibot", 0):.4f}',
                'koleo': f'{losses.get("koleo", 0):.4f}',
                'temp': f'{self.dino_loss.teacher_temp:.3f}',
                'gpu': f'{get_gpu_memory_gb():.1f}GB'
            })
        
        self.scheduler.step()
        
        # Epoch summary
        avg_loss = self.epoch_loss / self.steps_per_epoch
        avg_batch_time = np.mean(batch_times)
        throughput = self.cfg.batch_size / avg_batch_time
        
        log_epoch(self.epoch_csv, epoch, avg_loss,
                 self.scheduler.get_last_lr()[0], throughput, avg_batch_time)
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch} COMPLETE")
        print(f"  Avg loss: {avg_loss:.6f}")
        print(f"  LR: {self.scheduler.get_last_lr()[0]:.2e}")
        print(f"  Throughput: {throughput:.1f} samples/sec")
        print(f"  GPU: {get_gpu_memory_gb():.1f}GB")
        print(f"{'='*80}")
    
    def _train_step(self, crops, global_step):
        """Single training step"""
        crops = [c.to(self.device, non_blocking=True) for c in crops]
        global_crops = crops[:2]
        all_crops = crops
        
        # Check if heads are frozen
        heads_frozen = global_step < self.cfg.head_frozen_iters
        if heads_frozen:
            self.head_s.eval()
            for p in self.head_s.parameters():
                p.requires_grad = False
        else:
            self.head_s.train()
            for p in self.head_s.parameters():
                p.requires_grad = True
        
        with autocast('cuda', enabled=self.cfg.use_mixed_precision):
            # Student forward on all crops
            student_outputs = []
            for crop in all_crops:
                h = self.student(crop)
                z = self.head_s(h)
                p = self.predictor(z)
                student_outputs.append(p)
            
            # Teacher forward on global crops only
            with torch.no_grad():
                teacher_outputs = []
                for crop in global_crops:
                    h = self.teacher(crop)
                    z = self.head_t(h)
                    teacher_outputs.append(z)
            
            # DINO loss
            loss_dino = self.dino_loss(student_outputs, teacher_outputs)
            
            # iBOT loss (on first global crop with masking)
            loss_ibot = torch.tensor(0.0, device=self.device)
            if self.cfg.ibot_weight > 0:
                # Random mask ratio between min and max
                mask_ratio = np.random.uniform(
                    self.cfg.ibot_mask_ratio_min,
                    self.cfg.ibot_mask_ratio_max
                )
                
                # Get patch features (before projection head)
                student_h = self.student(global_crops[0])
                teacher_h = self.teacher(global_crops[0])
                
                # Apply head to patches
                B, N, D = student_h.shape if len(student_h.shape) == 3 else (student_h.shape[0], 1, student_h.shape[1])
                if len(student_h.shape) == 2:
                    student_h = student_h.unsqueeze(1)
                    teacher_h = teacher_h.unsqueeze(1)
                
                student_patches = self.head_s(student_h.reshape(-1, D)).reshape(B, N, -1)
                teacher_patches = self.head_t(teacher_h.reshape(-1, D)).reshape(B, N, -1)
                
                # Create mask and compute iBOT loss
                mask = random_masking(student_patches, mask_ratio)
                loss_ibot = self.ibot_loss(student_patches, teacher_patches, mask)
            
            # Koleo regularization
            loss_koleo = torch.tensor(0.0, device=self.device)
            if self.cfg.koleo_weight > 0:
                loss_koleo = self.koleo_loss(student_outputs[0])
            
            # Total loss
            total_loss = (loss_dino + 
                         self.cfg.ibot_weight * loss_ibot + 
                         self.cfg.koleo_weight * loss_koleo)
        
        # Backward
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return {
            'total': total_loss.item(),
            'dino': loss_dino.item(),
            'ibot': loss_ibot.item() if isinstance(loss_ibot, torch.Tensor) else 0,
            'koleo': loss_koleo.item() if isinstance(loss_koleo, torch.Tensor) else 0,
        }

