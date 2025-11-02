from torchvision import transforms


class MultiCropAugmentation:
    """
    Multi-crop augmentation as in DINOv2 / ExPLoRA:
    - 2 global crops (>32% of image, resized to global_size)
    - N local crops (5-32% of image, resized to local_size)
    """
    
    def __init__(self, global_size=224, local_size=98, n_local=8):
        # Global crop: 32-100% of image
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=(0.32, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Local crop: 5-32% of image
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=(0.05, 0.32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.n_local = n_local
    
    def __call__(self, img):
        """Returns list of [global_1, global_2, local_1, ..., local_N]"""
        crops = []
        crops.append(self.global_transform(img))
        crops.append(self.global_transform(img))
        for _ in range(self.n_local):
            crops.append(self.local_transform(img))
        return crops

