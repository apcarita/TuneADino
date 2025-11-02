from .dino import DINOLoss, sinkhorn_knopp
from .ibot import iBOTLoss, random_masking
from .regularization import KoleoLoss

__all__ = [
    'DINOLoss',
    'sinkhorn_knopp',
    'iBOTLoss',
    'random_masking',
    'KoleoLoss',
]

