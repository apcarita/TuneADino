# ExPLoRA Training

Parameter-efficient fine-tuning of DINOv3 using ExPLoRA (DINO + iBOT + LoRA).

```bash
cd Training
python train.py
```

Configuration is in `config.json`.

Efficient self-supervised adaptation:
1. **DINO** - Self-distillation with Sinkhorn-Knopp centering
2. **iBOT** - Masked patch prediction (BERT-style)
3. **LoRA** - Low-rank adaptation (parameter efficient)

Plus Koleo regularization for feature diversity.

## Architecture

- ViT-Large backbone with LoRA on frozen blocks
- Last 2 blocks fully unfrozen 
- DINO-iBOT shared head: 1024 → 256 → 2048 → 65536
- Student predictor for asymmetry


## Files

See `STRUCTURE.md` for detailed file organization.

## Configuration

All settings in `config.json`:
- `data_dirs`: List of dataset paths
- `pretrained_model`: Path to DINOv3 checkpoint
- `output_dir`: Where to save metrics
- `checkpoint_dir`: Where to save model checkpoints
- `training`: All hyperparameters

## Outputs

- `output/explora/batch_metrics.csv` - Per-batch metrics
- `output/explora/epoch_metrics.csv` - Per-epoch summary
- `checkpoints/explora/checkpoint_epoch_*.pth` - Model checkpoints

## Requirements

- PyTorch with CUDA
- timm (for ViT models)
- torchvision
- tqdm

## Notes

- Old implementation backed up to `_old_implementation/`
- All files under 400 lines (clean, modular code)
- Print statements for debugging (no silent failures)
- Automatic checkpoint resuming

