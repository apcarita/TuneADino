from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
from pathlib import Path


class MultiCropDataset(Dataset):
    """Dataset with multi-crop augmentation"""
    
    def __init__(self, roots, transform=None):
        self.samples = []
        
        print("Loading datasets:")
        for root in roots:
            root_path = Path(root)
            if not root_path.exists():
                print(f"  WARNING: {root} does not exist, skipping")
                continue
            
            try:
                folder = ImageFolder(str(root_path))
                self.samples.extend(folder.samples)
                print(f"  Loaded {len(folder.samples):,} images from {root}")
            except Exception as e:
                print(f"  WARNING: Failed to load {root}: {e}")
        
        if len(self.samples) == 0:
            raise ValueError("No images found in any data directory")
        
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
            print(f"WARNING: Error loading {path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

