import argparse
import os
import torch
import torchvision.transforms as T
from PIL import Image

from models import MODEL_REGISTRY
from utils.config import load_config, Config

def get_device(target_device="auto"):
    if target_device != "auto":
        return torch.device(target_device)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def print_embedding_diagnostics(z_c, z_n):
    """
    Prints channel-by-channel statistics for all 32 latent embedding dimensions.
    """
    total_dim = z_c.shape[1] + z_n.shape[1]
    print("\n" + "=" * 80)
    print(f"{'LATENT EMBEDDING DIAGNOSTICS':^80}")
    print(f"Total Dimensions: {total_dim} | Content Channels: {z_c.shape[1]} | Noise Channels: {z_n.shape[1]}")
    print(f"Spatial Feature Map Resolution: {z_c.shape[2]} x {z_c.shape[3]}")
    print("=" * 80)
    print(f"{'Index':<7} | {'Subspace':<10} | {'Mean':<12} | {'Std Dev':<12} | {'Min':<12} | {'Max':<12}")
    print("-" * 80)

    # 1. Content Embeddings (z_c)
    for c in range(z_c.shape[1]):
        channel = z_c[0, c, :, :]
        print(
            f"Dim {c:<3} | {'Content':<10} | "
            f"{channel.mean().item():<+12.5f} | "
            f"{channel.std().item():<12.5f} | "
            f"{channel.min().item():<+12.5f} | "
            f"{channel.max().item():<+12.5f}"
        )

    print("-" * 80)

    # 2. Noise Embeddings (z_n)
    offset = z_c.shape[1]
    for n in range(z_n.shape[1]):
        channel = z_n[0, n, :, :]
        print(
            f"Dim {offset + n:<3} | {'Noise':<10} | "
            f"{channel.mean().item():<+12.5f} | "
            f"{channel.std().item():<12.5f} | "
            f"{channel.min().item():<+12.5f} | "
            f"{channel.max().item():<+12.5f}"
        )
    print("=" * 80 + "\n")

def infer(args):
    device = get_device(args.device)
    print(f"--> [Hardware] Running inference on device: {device}")

    # Load configuration parameters if provided, else use defaults
    content_dim = 28
    noise_dim = 4
    if args.config and os.path.exists(args.config):
        cfg = load_config(args.config)
        if hasattr(cfg.model, "params"):
            content_dim = getattr(cfg.model.params, "content_dim", 28)
            noise_dim = getattr(cfg.model.params, "noise_dim", 4)

    # Instantiate model and load trained weights
    model = MODEL_REGISTRY.build(
        args.model,
        in_channels=3,
        out_channels=3,
        content_dim=content_dim,
        noise_dim=noise_dim
    ).to(device)

    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print(f"--> Loaded checkpoint: {args.weights}")

    # Load input image
    img = Image.open(args.input_image).convert("RGB")
    input_tensor = T.ToTensor()(img).unsqueeze(0).to(device)

    base_name, ext = os.path.splitext(args.output_image)
    if not ext:
        ext = ".png"

    with torch.no_grad():
        # Step 1: Encode into Content (z_c) and Noise (z_n) latent embeddings
        z_c, z_n = model.encode(input_tensor)

        # Step 2: Print diagnostic statistics for all 32 channels
        print_embedding_diagnostics(z_c, z_n)

        # Step 3: Decode noise using the dedicated Noise Decoder (D_noise)
        predicted_noise = model.decode_noise(z_n)

        # Step 4: Subtract predicted noise from the original input image to reconstruct clean output
        clean_by_subtraction = torch.clamp(input_tensor - predicted_noise, 0.0, 1.0)

        # Step 5: Direct clean decode via D_clean for auxiliary validation
        direct_clean_decoded = torch.clamp(model.decode_clean(z_c), 0.0, 1.0)

    # Save original input image fed to the network
    input_save_path = f"{base_name}_input_fed{ext}"
    T.ToPILImage()(input_tensor.squeeze(0).cpu()).save(input_save_path)
    print(f"--> Saved Input Image:               {input_save_path}")

    # Save reconstructed clean image obtained via (Input - Predicted_Noise)
    clean_sub_img = T.ToPILImage()(clean_by_subtraction.squeeze(0).cpu())
    clean_sub_img.save(args.output_image)
    print(f"--> Saved Subtracted Clean Image:    {args.output_image}")

    # Save raw noise decoder output (offset by 0.5 to visualize positive & negative residuals)
    noise_normalized = torch.clamp(predicted_noise.squeeze(0).cpu() + 0.5, 0.0, 1.0)
    noise_save_path = f"{base_name}_predicted_noise{ext}"
    T.ToPILImage()(noise_normalized).save(noise_save_path)
    print(f"--> Saved Noise Decoder Dump:        {noise_save_path}")

    # Save Clean Decoder direct output (x_hat = D_clean(z_c))
    direct_clean_path = f"{base_name}_direct_clean{ext}"
    T.ToPILImage()(direct_clean_decoded.squeeze(0).cpu()).save(direct_clean_path)
    print(f"--> Saved Direct Clean Decoder Dump: {direct_clean_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Disentangled Denoising Network")
    parser.add_argument("--model", type=str, default="disentangle_net", help="Registered model name")
    parser.add_argument("--config", type=str, default="configs/config_disentangle.yaml", help="Path to config YAML")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input noisy or clean image")
    parser.add_argument("--output_image", type=str, default="denoised_subtracted.png", help="Path for clean output")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "xpu", "cuda", "cpu"])
    args = parser.parse_args()

    infer(args)