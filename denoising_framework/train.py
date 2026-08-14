import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

    # Load YAML Configuration
    cfg = load_config(args.config)

    # CLI Overrides
    device_name = args.device if args.device else cfg.hardware.device
    epochs = args.epochs if args.epochs else cfg.training.epochs

    device = get_device(device_name)
    os.makedirs(cfg.experiment.output_dir, exist_ok=True)

    print(f"--> [Experiment] Launching: {cfg.experiment.name}")
    print(f"--> [Model] Building '{cfg.model.name}' via Registry...")

    # Pass dynamic model hyper-parameters if present
    model_kwargs = getattr(cfg.model, "params", Config({})).__dict__ if hasattr(cfg.model, "params") else {}
    model = MODEL_REGISTRY.build(cfg.model.name, **model_kwargs).to(device)

    # Data Loading
    train_dataset = DenoisingDataset(
        root_dir=cfg.dataset.train_dir, 
        patch_size=cfg.dataset.patch_size, 
        sigma=cfg.dataset.sigma, 
        is_train=True
    )
    val_dataset = DenoisingDataset(
        root_dir=cfg.dataset.val_dir, 
        sigma=cfg.dataset.sigma, 
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True, 
        num_workers=cfg.training.num_workers
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    criterion = nn.L1Loss() if cfg.training.loss == "L1" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    best_psnr = 0.0
    save_path = os.path.join(cfg.experiment.output_dir, f"best_{cfg.model.name}.pth")

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_psnr = evaluate(model, val_loader, device)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {loss:.4f} | Val PSNR: {val_psnr:.2f} dB")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), save_path)
            print(f"    [Checkpoint] Saved best weights -> {save_path}")

if __name__ == "__main__":
    main()