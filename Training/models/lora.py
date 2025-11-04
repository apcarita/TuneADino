import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer: W + (B @ A) * scaling"""
    
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        nn.init.kaiming_uniform_(self.lora_A, a=1)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x, original_output):
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return original_output + lora_out


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
        
        qkv_lora = LoRALayer(attn.qkv.in_features, attn.qkv.out_features, rank, alpha)
        proj_lora = LoRALayer(attn.proj.in_features, attn.proj.out_features, rank, alpha)
        
        # Save original forward methods
        qkv_orig_forward = attn.qkv.forward
        proj_orig_forward = attn.proj.forward
        
        def make_lora_forward(orig_forward, lora_layer):
            def forward(x):
                return lora_layer(x, orig_forward(x))
            return forward
        
        attn.qkv.forward = make_lora_forward(qkv_orig_forward, qkv_lora)
        attn.proj.forward = make_lora_forward(proj_orig_forward, proj_lora)
        
        lora_params.extend([qkv_lora.lora_A, qkv_lora.lora_B])
        lora_params.extend([proj_lora.lora_A, proj_lora.lora_B])
        
        block.qkv_lora = qkv_lora
        block.proj_lora = proj_lora
    
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

