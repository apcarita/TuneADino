import torch
import torch.nn as nn
import timm
from pathlib import Path
from .lora import apply_lora_to_vit
from .heads import create_dino_ibot_head, create_predictor


def create_explora_models(
    img_size=224,
    pretrained_path=None,
    checkpoint_data=None,
    device='cuda',
    lora_rank=8,
    lora_alpha=16,
    unfreeze_last_n_blocks=2,
    drop_path_rate=0.2,
    gradient_checkpointing=False,
    bottleneck_dim=256,
    hidden_dim=2048,
    out_dim=65536
):
    """Create ExPLoRA student/teacher models with LoRA and DINO-iBOT heads"""
    
    # Student with drop-path
    student = timm.create_model(
        'vit_large_patch16_224',
        pretrained=False,
        num_classes=0,
        img_size=img_size,
        dynamic_img_size=True,
        drop_path_rate=drop_path_rate
    )
    
    # Load pretrained weights (DINOv3) if starting fresh
    if checkpoint_data is None and pretrained_path and Path(pretrained_path).exists():
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        # Handle DINOv3 checkpoint format (may be wrapped)
        if isinstance(checkpoint, dict):
            # Try common keys: 'model', 'state_dict', or direct state_dict
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # Assume it's the state_dict itself
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        student.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {pretrained_path}")
        del checkpoint
        torch.cuda.empty_cache()
    
    # Apply LoRA to student (must be done before loading checkpoint that contains LoRA)
    student, lora_params = apply_lora_to_vit(
        student, rank=lora_rank, alpha=lora_alpha, 
        unfreeze_last_n_blocks=unfreeze_last_n_blocks
    )
    
    # Enable gradient checkpointing if requested (saves memory at cost of compute)
    if gradient_checkpointing:
        student.set_grad_checkpointing(enable=True)
        print("Enabled gradient checkpointing for memory efficiency")
    
    # Load student from checkpoint if resuming (checkpoint contains LoRA params)
    if checkpoint_data is not None:
        student.load_state_dict(checkpoint_data['student'], strict=False)
        print("Loaded student from checkpoint")
    
    # Teacher (no drop-path for stability) - MUST have same LoRA architecture as student
    teacher = timm.create_model(
        'vit_large_patch16_224',
        pretrained=False,
        num_classes=0,
        img_size=img_size,
        dynamic_img_size=True,
        drop_path_rate=0.0
    )
    
    # Apply LoRA to teacher (same architecture as student)
    teacher, _ = apply_lora_to_vit(
        teacher, rank=lora_rank, alpha=lora_alpha,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks
    )
    
    # Teacher doesn't need gradient checkpointing (no gradients computed)
    if gradient_checkpointing:
        teacher.set_grad_checkpointing(enable=False)
    
    # Load teacher weights
    if checkpoint_data is not None:
        teacher.load_state_dict(checkpoint_data['teacher'], strict=False)
        print("Loaded teacher from checkpoint")
    else:
        teacher.load_state_dict(student.state_dict(), strict=False)
        print("Initialized teacher from student")
    
    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    
    print(f"  Base size: {img_size}x{img_size} (dynamic enabled)")
    print(f"  Drop-path: Student={drop_path_rate}, Teacher=0.0")
    
    # Create DINO-iBOT shared heads
    head_student = create_dino_ibot_head(1024, bottleneck_dim, hidden_dim, out_dim).to(device)
    head_teacher = create_dino_ibot_head(1024, bottleneck_dim, hidden_dim, out_dim).to(device)
    
    # Create predictor (student only)
    predictor = create_predictor(out_dim, hidden_dim=8192).to(device)
    
    if checkpoint_data is not None:
        head_student.load_state_dict(checkpoint_data['head_student'])
        head_teacher.load_state_dict(checkpoint_data['head_teacher'])
        predictor.load_state_dict(checkpoint_data['predictor'])
        print("Loaded heads and predictor from checkpoint")
    else:
        head_teacher.load_state_dict(head_student.state_dict())
        print("Initialized DINO-iBOT heads (bottleneck={}, hidden={}, out={})".format(
            bottleneck_dim, hidden_dim, out_dim))
    
    head_teacher.eval()
    for p in head_teacher.parameters():
        p.requires_grad = False
    
    return student, teacher, head_student, head_teacher, predictor, lora_params

