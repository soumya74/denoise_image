import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T  # Added this import
import random

class DenoisingDataset(Dataset):
    def __init__(self, root_dir, patch_size=128, sigma=25.0, is_train=True):
        self.root_dir = root_dir
        self.clean_dir = os.path.join(root_dir, "clean")
        self.noisy_dir = os.path.join(root_dir, "noisy")
        self.patch_size = patch_size
        self.is_train = is_train
        
        # We accept 'sigma' to match the kwargs passed by train.py, 
        # but we don't use it here since your noise is pre-computed.

        # Verify directories exist to prevent silent failures
        if not os.path.exists(self.clean_dir) or not os.path.exists(self.noisy_dir):
            raise FileNotFoundError(f"Ensure both 'clean' and 'noisy' folders exist inside {root_dir}")

        # Get list of matching image filenames
        self.filenames = sorted([
            f for f in os.listdir(self.clean_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        clean_path = os.path.join(self.clean_dir, filename)
        noisy_path = os.path.join(self.noisy_dir, filename)

        clean_img = Image.open(clean_path).convert('RGB')
        noisy_img = Image.open(noisy_path).convert('RGB')

        clean_tensor = TF.to_tensor(clean_img)
        noisy_tensor = TF.to_tensor(noisy_img)

        # Synchronized Data Augmentation (Train Mode)
        if self.is_train:
            # Random Crop (must crop the exact same spatial region from both images)
            # CHANGED: TF.RandomCrop to T.RandomCrop
            i, j, h, w = T.RandomCrop.get_params(
                clean_tensor, output_size=(self.patch_size, self.patch_size)
            )
            clean_tensor = TF.crop(clean_tensor, i, j, h, w)
            noisy_tensor = TF.crop(noisy_tensor, i, j, h, w)

            # Random Horizontal Flip
            if random.random() > 0.5:
                clean_tensor = TF.hflip(clean_tensor)
                noisy_tensor = TF.hflip(noisy_tensor)

            # Random Vertical Flip
            if random.random() > 0.5:
                clean_tensor = TF.vflip(clean_tensor)
                noisy_tensor = TF.vflip(noisy_tensor)

        return noisy_tensor, clean_tensor