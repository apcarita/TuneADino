import torch.nn as nn


def create_dino_ibot_head(in_dim=1024, bottleneck_dim=256, hidden_dim=2048, out_dim=65536):
    """
    Create DINO-iBOT shared head as in ExPLoRA paper:
    3-layer MLP with bottleneck (256), hidden (2048), output (65536)
    """
    return nn.Sequential(
        nn.Linear(in_dim, bottleneck_dim),
        nn.GELU(),
        nn.Linear(bottleneck_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


def create_predictor(in_dim=65536, hidden_dim=8192):
    """
    Student-only predictor for asymmetry (BYOL-style)
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, in_dim)
    )

