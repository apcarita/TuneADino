import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation: y = Wx + (B @ A @ x) * scaling
    Wraps the original linear layer for proper DataParallel support
    """
    
    def __init__(self, original_linear, rank=8, alpha=16):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        nn.init.kaiming_uniform_(self.lora_A, a=1)
        nn.init.zeros_(self.lora_B)
        
        # Freeze original weights
        for param in self.original_linear.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Original linear transformation
        result = self.original_linear(x)
        # Add LoRA adaptation
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result + lora_out


def apply_lora_to_vit(model, rank=8, alpha=16, unfreeze_last_n_blocks=2):
    """
    Apply LoRA to ViT following ExPLoRA:
    - Unfreeze last N blocks completely
    - Apply LoRA to attention layers in frozen blocks
    """
    for param in model.parameters():
        param.requires_grad = False
    
    n_blocks = len(model.blocks)
    freeze_until = n_blocks - unfreeze_last_n_blocks
    
    print(f"\nApplying ExPLoRA:")
    print(f"  Total blocks: {n_blocks}")
    print(f"  Fully unfrozen: blocks {freeze_until}-{n_blocks-1}")
    print(f"  LoRA applied: blocks 0-{freeze_until-1} (rank={rank}, alpha={alpha})")
    
    lora_params = []
    
    # Apply LoRA to frozen blocks
    for block_idx in range(freeze_until):
        block = model.blocks[block_idx]
        attn = block.attn
        
        # Replace qkv and proj with LoRA wrappers
        qkv_lora = LoRALinear(attn.qkv, rank, alpha)
        proj_lora = LoRALinear(attn.proj, rank, alpha)
        
        attn.qkv = qkv_lora
        attn.proj = proj_lora
        
        lora_params.extend([qkv_lora.lora_A, qkv_lora.lora_B])
        lora_params.extend([proj_lora.lora_A, proj_lora.lora_B])
    
    # Unfreeze last N blocks
    for block_idx in range(freeze_until, n_blocks):
        for param in model.blocks[block_idx].parameters():
            param.requires_grad = True
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_param_count = sum(p.numel() for p in lora_params)
    
    print(f"\nParameter efficiency:")
    print(f"  Total: {total_params:,} ({100*trainable_params/total_params:.2f}% trainable)")
    print(f"  LoRA: {lora_param_count:,}")
    print(f"  Unfrozen blocks: {trainable_params - lora_param_count:,}")
    
    return model, lora_params

