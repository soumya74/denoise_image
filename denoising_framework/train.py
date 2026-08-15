import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from models import MODEL_REGISTRY
from dataset import DenoisingDataset
from engine import train_one_epoch, evaluate
from utils.config import load_config, Config

def get_device(target_device="auto"):
    if target_device != "auto":
        return torch.device(target_device)
    
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        print(f"--> [Hardware] Using Intel XPU Accelerator: {torch.xpu.get_device_name(0)}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"--> [Hardware] Using NVIDIA CUDA Accelerator: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("--> [Hardware] No GPU/XPU accelerator detected. Running on CPU.")
    return device

def main():
    parser = argparse.ArgumentParser(description="Modular Image Denoising Trainer")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config file")
    parser.add_argument("--device", type=str, default=None, help="Override hardware device setting")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epoch count")
    args = parser.parse_args()

    cfg = load_config(args.config)

    device_name = args.device if args.device else cfg.hardware.device
    epochs = args.epochs if args.epochs else cfg.training.epochs

    device = get_device(device_name)
    os.makedirs(cfg.experiment.output_dir, exist_ok=True)

    print(f"--> [Experiment] Launching: {cfg.experiment.name}")
    print(f"--> [Model] Building '{cfg.model.name}' via Registry...")

    model_kwargs = getattr(cfg.model, "params", Config({})).__dict__ if hasattr(cfg.model, "params") else {}
    model = MODEL_REGISTRY.build(cfg.model.name, **model_kwargs).to(device)

    # --- Dataset Loading & Dynamic Splitting ---
    # Load dataset twice: one with training augmentations, one without
    full_train_ds = DenoisingDataset(cfg.dataset.train_dir, patch_size=cfg.dataset.patch_size, is_train=True)
    full_test_ds = DenoisingDataset(cfg.dataset.train_dir, is_train=False)

    num_items = len(full_train_ds)
    test_split = getattr(cfg.dataset, "test_split", 0.1)
    test_size = int(num_items * test_split)
    train_size = num_items - test_size

    # Generate random indices for the split
    indices = torch.randperm(num_items).tolist()
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    # Create subsets
    train_dataset = Subset(full_train_ds, train_indices)
    test_dataset = Subset(full_test_ds, test_indices)

    print(f"--> [Data] Total images: {num_items} | Train: {train_size} | Test: {test_size}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True, 
        num_workers=cfg.training.num_workers
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    criterion = nn.L1Loss() if cfg.training.loss == "L1" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # --- Metric Tracking ---
    history_train_loss = []
    history_test_loss = []
    history_test_psnr = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_psnr = evaluate(model, test_loader, criterion, device)
        
        history_train_loss.append(train_loss)
        history_test_loss.append(test_loss)
        history_test_psnr.append(test_psnr)
        
        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test PSNR: {test_psnr:.2f} dB")

    # --- Final Checkpoint Save ---
    save_path = os.path.join(cfg.experiment.output_dir, f"final_{cfg.model.name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"--> [Checkpoint] Saved final model weights -> {save_path}")

    # --- Plotting Curves ---
    plt.figure(figsize=(14, 5))

    # Plot 1: Losses
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history_train_loss, label='Train Loss', marker='o')
    plt.plot(range(1, epochs + 1), history_test_loss, label='Test Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Test Loss')
    plt.grid(True)
    plt.legend()

    # Plot 2: PSNR
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history_test_psnr, label='Test PSNR', color='green', marker='^')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.title('Test PSNR Accuracy')
    plt.grid(True)
    plt.legend()

    plot_path = os.path.join(cfg.experiment.output_dir, "training_curves.png")
    plt.savefig(plot_path)
    print(f"--> [Plot] Saved training curves to -> {plot_path}")

if __name__ == "__main__":
    main()