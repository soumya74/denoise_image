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

    train_loader = DataLoader(
        Subset(full_train_ds, train_indices),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers
    )
    test_loader = DataLoader(
        Subset(full_test_ds, test_indices),
        batch_size=1,
        shuffle=False
    )

    l1_loss = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    w = cfg.loss_weights

    # History Trackers for all 7 individual losses + Total Loss + PSNR
    history = {
        "total": [],
        "clean_rec": [],
        "clean_self": [],
        "clean_cross": [],
        "noisy_rec": [],
        "noise_rec": [],
        "latent": [],
        "noise_zero": [],
        "val_psnr": []
    }

    epochs = cfg.training.epochs
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = {
            "total": 0.0,
            "clean_rec": 0.0,
            "clean_self": 0.0,
            "clean_cross": 0.0,
            "noisy_rec": 0.0,
            "noise_rec": 0.0,
            "latent": 0.0,
            "noise_zero": 0.0
        }

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            gt_noise = noisy - clean

            optimizer.zero_grad()
            res = model(noisy_img=noisy, clean_img=clean)

            # 1. Clean Reconstruction Loss: ||x_hat - x||_1
            l_clean_rec = l1_loss(res["x_hat"], clean)

            # 2. Clean Self-Reconstruction Loss: ||x_hat_clean - x||_1
            l_clean_self = l1_loss(res["x_hat_clean"], clean)

            # 3. Cross-Reconstruction Consistency Loss: ||x_hat - x_hat_clean||_1
            l_clean_cross = l1_loss(res["x_hat"], res["x_hat_clean"].detach())

            # 4. Noisy Image Reconstruction Loss: ||y_hat - y||_1
            l_noisy_rec = l1_loss(res["y_hat"], noisy)

            # 5. Explicit Noise Reconstruction Loss: ||n_hat - n||_1
            l_noise_rec = l1_loss(res["n_hat"], gt_noise)

            # 6. Latent Content Consistency Loss: ||z_c^y - z_c^x||_1
            l_content_latent = l1_loss(res["z_c_y"], res["z_c_x"].detach())

            # 7. Clean Latent Noise Suppression Loss: ||z_n^x||_1 -> 0
            l_noise_zero = torch.mean(torch.abs(res["z_n_x"]))

            total_loss = (
                w.clean_rec * l_clean_rec +
                w.clean_self * l_clean_self +
                getattr(w, "clean_cross", 1.0) * l_clean_cross +
                w.noisy_rec * l_noisy_rec +
                w.noise_rec * l_noise_rec +
                w.content_latent * l_content_latent +
                getattr(w, "noise_zero", 1.0) * l_noise_zero
            )

            total_loss.backward()
            
            # --- THE EXPLOSION SHIELD ---
            # Prevents gradients from exceeding a safe magnitude (1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            epoch_losses["total"] += total_loss.item()
            epoch_losses["clean_rec"] += l_clean_rec.item()
            epoch_losses["clean_self"] += l_clean_self.item()
            epoch_losses["clean_cross"] += l_clean_cross.item()
            epoch_losses["noisy_rec"] += l_noisy_rec.item()
            epoch_losses["noise_rec"] += l_noise_rec.item()
            epoch_losses["latent"] += l_content_latent.item()
            epoch_losses["noise_zero"] += l_noise_zero.item()

        n_batches = len(train_loader)
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
            history[k].append(epoch_losses[k])

        # Validation PSNR Evaluation
        model.eval()
        total_psnr = 0.0
        with torch.no_grad():
            for test_noisy, test_clean in test_loader:
                test_noisy, test_clean = test_noisy.to(device), test_clean.to(device)
                z_c, _ = model.encode(test_noisy)
                denoised = torch.clamp(model.decode_clean(z_c), 0.0, 1.0)
                total_psnr += calculate_psnr(denoised, test_clean)
        val_psnr = total_psnr / len(test_loader)
        history["val_psnr"].append(val_psnr)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | Total: {epoch_losses['total']:.4f} | "
            f"CleanRec: {epoch_losses['clean_rec']:.4f}, SelfClean: {epoch_losses['clean_self']:.4f}, "
            f"CrossClean: {epoch_losses['clean_cross']:.4f}, NoisyRec: {epoch_losses['noisy_rec']:.4f}, "
            f"NoiseRec: {epoch_losses['noise_rec']:.4f}, Latent: {epoch_losses['latent']:.4f}, "
            f"NoiseZero: {epoch_losses['noise_zero']:.4f} | PSNR: {val_psnr:.2f} dB"
        )

    # Save Checkpoint
    save_path = os.path.join(cfg.experiment.output_dir, f"final_{cfg.model.name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"--> [Model] Saved checkpoint to: {save_path}")

    # Plot Diagnostics in a 3x3 Subplot Grid
    plt.figure(figsize=(18, 14))
    plt.suptitle(f"Disentanglement Training Diagnostics: 7 Losses & PSNR ({cfg.experiment.name})", fontsize=16, fontweight='bold')
    epoch_axis = range(1, epochs + 1)

    loss_plots = [
        ("Clean Rec: ||x̂ - x||₁", history["clean_rec"], "tab:blue"),
        ("Clean Self: ||x̂_clean - x||₁", history["clean_self"], "tab:purple"),
        ("Clean Cross: ||x̂ - x̂_clean||₁", history["clean_cross"], "tab:cyan"),
        ("Noisy Rec: ||ŷ - y||₁", history["noisy_rec"], "tab:orange"),
        ("Noise Rec: ||n̂ - n||₁", history["noise_rec"], "tab:red"),
        ("Noise Zero: ||z_n^x||₁ → 0", history["noise_zero"], "tab:pink"),
        ("Latent Content: ||z_c^y - z_c^x||₁", history["latent"], "tab:brown"),
        ("Total Weighted Loss", history["total"], "black"),
        ("Validation PSNR (dB)", history["val_psnr"], "tab:green")
    ]

    for idx, (title, data, color) in enumerate(loss_plots, 1):
        plt.subplot(3, 3, idx)
        marker = '^' if "PSNR" in title else 'o'
        plt.plot(epoch_axis, data, marker=marker, color=color, linewidth=1.8, label=title.split(":")[0])
        plt.xlabel("Epochs", fontsize=10)
        plt.ylabel("Value", fontsize=10)
        plt.title(title, fontsize=11, fontweight='semibold')
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="best", fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    plot_path = os.path.join(cfg.experiment.output_dir, "training_losses_disentangle.png")
    plt.savefig(plot_path, dpi=200)
    print(f"--> [Plot] Saved complete 3x3 diagnostics plot to: {plot_path}")

if __name__ == "__main__":
    main()