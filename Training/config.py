import json
import sys
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    weight_decay: float
    total_epochs: int
    warmup_epochs: int
    
    global_crop_size: int
    local_crop_size: int
    n_local_crops: int
    
    teacher_temp: float
    student_temp: float
    teacher_temp_final: float
    teacher_temp_warmup_epochs: int
    
    ema_momentum_initial: float
    ema_momentum_final: float
    
    use_mixed_precision: bool
    head_frozen_iters: int
    gradient_checkpointing: bool
    
    lora_rank: int
    lora_alpha: int
    unfreeze_last_n_blocks: int
    drop_path_rate: float
    
    koleo_weight: float
    ibot_mask_ratio_min: float
    ibot_mask_ratio_max: float
    ibot_weight: float


def load_config(config_path='config.json'):
    """Load and validate configuration"""
    config_file = Path(config_path)
    
    # If not found, try relative to this script's directory
    if not config_file.exists():
        script_dir = Path(__file__).parent
        config_file = script_dir / config_path
    
    if not config_file.exists():
        print(f"ERROR: Config file not found: {config_path}")
        print(f"Tried: {Path(config_path).absolute()}")
        print(f"Tried: {config_file.absolute()}")
        print(f"\nPlease run from the Training directory:")
        print(f"  cd {Path(__file__).parent}")
        print(f"  python train.py")
        sys.exit(1)
    
    with open(config_file, 'r') as f:
        data = json.load(f)
    
    # Extract paths
    data_dirs = data['data_dirs']
    pretrained_model = data['pretrained_model']
    output_dir = Path(data['output_dir'])
    checkpoint_dir = Path(data['checkpoint_dir'])
    
    # Create training config
    t = data['training']
    training = TrainingConfig(
        batch_size=t['batch_size'],
        learning_rate=t['learning_rate'],
        weight_decay=t['weight_decay'],
        total_epochs=t['total_epochs'],
        warmup_epochs=t['warmup_epochs'],
        
        global_crop_size=t['global_crop_size'],
        local_crop_size=t['local_crop_size'],
        n_local_crops=t['n_local_crops'],
        
        teacher_temp=t['teacher_temp'],
        student_temp=t['student_temp'],
        teacher_temp_final=t['teacher_temp_final'],
        teacher_temp_warmup_epochs=t['teacher_temp_warmup_epochs'],
        
        ema_momentum_initial=t['ema_momentum_initial'],
        ema_momentum_final=t['ema_momentum_final'],
        
        use_mixed_precision=t['use_mixed_precision'],
        head_frozen_iters=t.get('head_frozen_iters', 3000),
        gradient_checkpointing=t.get('gradient_checkpointing', False),
        
        lora_rank=data['lora_rank'],
        lora_alpha=data['lora_alpha'],
        unfreeze_last_n_blocks=data['unfreeze_last_n_blocks'],
        drop_path_rate=data['drop_path_rate'],
        
        koleo_weight=t.get('koleo_weight', 0.1),
        ibot_mask_ratio_min=t.get('ibot_mask_ratio_min', 0.1),
        ibot_mask_ratio_max=t.get('ibot_mask_ratio_max', 0.5),
        ibot_weight=t.get('ibot_weight', 1.0),
    )
    
    return data_dirs, pretrained_model, output_dir, checkpoint_dir, training

