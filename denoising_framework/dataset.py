import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class DenoisingDataset(Dataset):
    def __init__(self, root_dir, patch_size=128, sigma=25, is_train=True):
        self.root_dir = root_dir
        self.sigma = sigma / 255.0  # Normalize AWGN std
        self.is_train = is_train
        
        self.image_paths = [
            os.path.join(root_dir, f) for f in os.listdir(root_dir) 
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if is_train:
            self.transform = T.Compose([
                T.RandomCrop(patch_size),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor()
            ])
        else:
            self.transform = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        clean_img = Image.open(self.image_paths[idx]).convert('RGB')
        clean_tensor = self.transform(clean_img)
        
        # Synthesize Additive White Gaussian Noise (AWGN)
        noise = torch.randn_like(clean_tensor) * self.sigma
        noisy_tensor = torch.clamp(clean_tensor + noise, 0.0, 1.0)
        
        return noisy_tensor, clean_tensor