#!/usr/bin/env python3
"""
ExPLoRA Training Script
DINO + iBOT + LoRA for parameter-efficient adaptation
"""
import torch
from pathlib import Path

from config import load_config
from training import ExPLoRATrainer


def main():
    # Load configuration
    print("Loading configuration...")
    data_dirs, pretrained_model, output_dir, checkpoint_dir, config = load_config('config.json')
    
    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        exit(1)
    
    device = torch.device('cuda')
    
    # Create trainer
    trainer = ExPLoRATrainer(
        config=config,
        data_dirs=data_dirs,
        pretrained_path=pretrained_model,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        device=device
    )
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()

