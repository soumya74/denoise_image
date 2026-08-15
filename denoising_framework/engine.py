import torch
import torch.nn as nn
from math import log10

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * log10(1.0 / torch.sqrt(mse).item())

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for noisy, clean in dataloader:
        noisy, clean = noisy.to(device), clean.to(device)
        
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, criterion, device): # Added criterion to arguments
    model.eval()
    total_psnr = 0.0
    total_loss = 0.0
    
    for noisy, clean in dataloader:
        noisy, clean = noisy.to(device), clean.to(device)
        output = model(noisy)
        
        # Calculate test loss
        loss = criterion(output, clean)
        total_loss += loss.item()
        
        # Calculate test PSNR
        output = torch.clamp(output, 0.0, 1.0)
        psnr = calculate_psnr(output, clean)
        total_psnr += psnr
        
    return total_loss / len(dataloader), total_psnr / len(dataloader)