import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from math import log10
import matplotlib.pyplot as plt

from models import MODEL_REGISTRY
from dataset import DenoisingDataset
from utils.config import load_config, Config

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * log10(1.0 / torch.sqrt(mse).item())

def get_device(target_device="auto"):
    if target_device != "auto":
        return torch.device(target_device)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        print(f"--> [Hardware] Using Intel XPU: {torch.xpu.get_device_name(0)}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"--> [Hardware] Using NVIDIA CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("--> [Hardware] Running on CPU.")
    return device

def main():
    parser = argparse.ArgumentParser(description="Dual-Decoder Disentanglement Trainer")
    parser.add_argument("--config", type=str, default="configs/config_disentangle.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.hardware.device)
    os.makedirs(cfg.experiment.output_dir, exist_ok=True)

    print(f"--> [Experiment] Initializing: {cfg.experiment.name}")
    model_kwargs = getattr(cfg.model, "params", Config({})).__dict__ if hasattr(cfg.model, "params") else {}
    model = MODEL_REGISTRY.build(cfg.model.name, **model_kwargs).to(device)

    # Data Loading
    full_train_ds = DenoisingDataset(cfg.dataset.train_dir, patch_size=cfg.dataset.patch_size, is_train=True)
    full_test_ds = DenoisingDataset(cfg.dataset.train_dir, is_train=False)

    num_items = len(full_train_ds)
    test_split = getattr(cfg.dataset, "test_split", 0.1)
    test_size = int(num_items * test_split)
    train_size = num_items - test_size

    indices = torch.randperm(num_items).tolist()
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    train_loader = DataLoader(Subset(full_train_ds, train_indices), batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers)
    test_loader = DataLoader(Subset(full_test_ds, test_indices), batch_size=1, shuffle=False)

    l1_loss = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    w = cfg.loss_weights
    history_total_loss, history_test_psnr = [], []

    epochs = cfg.training.epochs
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = {
            "total": 0.0, "clean_rec": 0.0, "noisy_rec": 0.0, 
            "noise_rec": 0.0, "clean_self": 0.0, "latent": 0.0, "noise_zero": 0.0
        }

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            # Calculate explicit ground truth noise: n = y - x
            gt_noise = noisy - clean

            optimizer.zero_grad()
            res = model(noisy_img=noisy, clean_img=clean)

            # --- THE 6 DISENTANGLEMENT LOSS FUNCTIONS ---
            # 1. Clean Reconstruction Loss: ||x_hat - x||_1
            l_clean_rec = l1_loss(res["x_hat"], clean)

            # 2. Noisy Image Reconstruction Loss: ||y_hat - y||_1 where y_hat = x_hat + n_hat
            l_noisy_rec = l1_loss(res["y_hat"], noisy)

            # 3. Explicit Noise Reconstruction Loss: ||n_hat - n||_1
            l_noise_rec = l1_loss(res["n_hat"], gt_noise)

            # 4. Clean Identity Reconstruction Loss: ||x_hat_clean - x||_1
            l_clean_self = l1_loss(res["x_hat_clean"], clean)

            # 5. Latent Content Consistency Loss: ||z_c^y - z_c^x||_1
            l_content_latent = l1_loss(res["z_c_y"], res["z_c_x"])

            # 6. Clean Latent Noise Suppression Loss: ||z_n^x||_1 (Forces 4-D noise latent to zero for clean images)
            l_noise_zero = torch.mean(torch.abs(res["z_n_x"]))

            # Total Weighted Objective
            total_loss = (
                w.clean_rec * l_clean_rec +
                w.noisy_rec * l_noisy_rec +
                w.noise_rec * l_noise_rec +
                w.clean_self * l_clean_self +
                w.content_latent * l_content_latent +
                w.noise_zero * l_noise_zero
            )

            total_loss = (
                w.clean_rec * l_clean_rec +
                w.noisy_rec * l_noisy_rec +
                w.noise_rec * l_noise_rec +
                w.clean_self * l_clean_self +
                w.content_latent * l_content_latent
            )

            total_loss.backward()
            optimizer.step()

            epoch_losses["total"] += total_loss.item()
            epoch_losses["clean_rec"] += l_clean_rec.item()
            epoch_losses["noisy_rec"] += l_noisy_rec.item()
            epoch_losses["noise_rec"] += l_noise_rec.item()
            epoch_losses["clean_self"] += l_clean_self.item()
            epoch_losses["latent"] += l_content_latent.item()
            epoch_losses["noise_zero"] += l_noise_zero.item()

        n_batches = len(train_loader)
        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        # Evaluate PSNR
        model.eval()
        total_psnr = 0.0
        with torch.no_grad():
            for test_noisy, test_clean in test_loader:
                test_noisy, test_clean = test_noisy.to(device), test_clean.to(device)
                z_c, _ = model.encode(test_noisy)
                denoised = torch.clamp(model.decode_clean(z_c), 0.0, 1.0)
                total_psnr += calculate_psnr(denoised, test_clean)
        val_psnr = total_psnr / len(test_loader)

        history_total_loss.append(epoch_losses["total"])
        history_test_psnr.append(val_psnr)

        print(
            f"Epoch [{epoch}/{epochs}] | Loss: {epoch_losses['total']:.4f} | "
            f"CleanRec: {epoch_losses['clean_rec']:.4f}, NoisyRec: {epoch_losses['noisy_rec']:.4f}, "
            f"NoiseRec: {epoch_losses['noise_rec']:.4f}, SelfRec: {epoch_losses['clean_self']:.4f}, "
            f"Latent: {epoch_losses['latent']:.4f} | Val PSNR: {val_psnr:.2f} dB"
        )

    # Save weights
    save_path = os.path.join(cfg.experiment.output_dir, f"final_{cfg.model.name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"--> Saved checkpoint to: {save_path}")

    # Plot Curves
    plt.figure(figsize=(12, 5))
    plt.suptitle(f"Disentanglement Network Training Summary", fontsize=14, fontweight='bold')
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history_total_loss, marker='o', label='Total Weighted Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Training Loss'); plt.grid(True); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history_test_psnr, marker='^', color='green', label='Test PSNR')
    plt.xlabel('Epochs'); plt.ylabel('PSNR (dB)'); plt.title('Validation PSNR (dB)'); plt.grid(True); plt.legend()

    plot_path = os.path.join(cfg.experiment.output_dir, "training_curves_disentangle.png")
    plt.savefig(plot_path)
    print(f"--> Saved curves to: {plot_path}")

if __name__ == "__main__":
    main()