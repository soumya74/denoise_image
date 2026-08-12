import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import MODEL_REGISTRY
from dataset import DenoisingDataset
from engine import train_one_epoch, evaluate

def main():
    parser = argparse.ArgumentParser(description="Modular Image Denoising Pipeline")
    parser.add_argument("--model", type=str, default="baseline_cnn", help="Model name registered in models/")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to train image folder")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to val image folder")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigma", type=float, default=25.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using device: {device}")

    # Build dynamically via Registry
    model = MODEL_REGISTRY.build(args.model).to(device)
    print(f"--> Initialized registered model: '{args.model}'")

    train_dataset = DenoisingDataset(args.train_dir, sigma=args.sigma, is_train=True)
    val_dataset = DenoisingDataset(args.val_dir, sigma=args.sigma, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_psnr = 0.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_psnr = evaluate(model, val_loader, device)
        print(f"Epoch [{epoch}/{args.epochs}] - Train Loss: {loss:.4f} | Val PSNR: {val_psnr:.2f} dB")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), f"best_{args.model}.pth")
            print(f"    Saved best checkpoint: best_{args.model}.pth")

if __name__ == "__main__":
    main()