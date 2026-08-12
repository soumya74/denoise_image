import argparse
import torch
import torchvision.transforms as T
from PIL import Image
from models import MODEL_REGISTRY

def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model structure dynamically
    model = MODEL_REGISTRY.build(args.model).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    img = Image.open(args.input_image).convert("RGB")
    transform = T.ToTensor()
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        output = torch.clamp(output, 0.0, 1.0)

    out_img = T.ToPILImage()(output.squeeze(0).cpu())
    out_img.save(args.output_image)
    print(f"Saved denoised image to: {args.output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--output_image", type=str, default="denoised.png")
    args = parser.parse_args()
    infer(args)