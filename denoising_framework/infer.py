import argparse
import torch
import torchvision.transforms as T
from PIL import Image
from models import MODEL_REGISTRY
from utils.config import load_config

def get_device(target_device="auto"):
    if target_device != "auto":
        return torch.device(target_device)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def infer(args):
    # Load config if specified, or fall back to CLI args
    device = get_device(args.device)
    print(f"--> Running inference on device: {device}")
    
    # Instantiate model from registry
    model = MODEL_REGISTRY.build(args.model).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # Load and preprocess input image
    img = Image.open(args.input_image).convert("RGB")
    transform = T.ToTensor()
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        output = torch.clamp(output, 0.0, 1.0)

    # Save output image
    out_img = T.ToPILImage()(output.squeeze(0).cpu())
    out_img.save(args.output_image)
    print(f"--> Denoised image saved to: {args.output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Denoising Inference Script")
    parser.add_argument("--model", type=str, required=True, help="Model name registered in models/")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained .pth checkpoint")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input noisy image")
    parser.add_argument("--output_image", type=str, default="denoised.png", help="Path to save result")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "xpu", "cuda", "cpu"])
    args = parser.parse_args()
    infer(args)