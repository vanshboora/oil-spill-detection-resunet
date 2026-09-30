import os
import argparse
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.resunet import LargeUNet


def predict_mask(
    image_path: str,
    checkpoint_path: str,
    output_path: str = "output_prediction.png",
    img_size: int = 256,
    threshold: float = 0.5,
    device: str = None
):
    """
    Loads a trained ResUNet model checkpoint, predicts oil spill segmentation mask
    for an input SAR image, and saves the visualized overlay.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # 1. Read SAR Image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    raw_image = cv2.imread(image_path)
    if raw_image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    original_h, original_w = rgb_image.shape[:2]

    # 2. Preprocess
    transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
    tensor_input = transform(image=rgb_image)["image"].unsqueeze(0).to(device)

    # 3. Load Checkpoint & Model
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    features = checkpoint.get("features", [64, 128, 256, 512, 1024])

    model = LargeUNet(in_channels=3, out_channels=1, features=features).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 4. Inference
    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    # 5. Post-process & threshold
    mask_binary = (probs > threshold).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask_binary, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    # 6. Create Visual Overlay (Red tint on detected oil spills)
    overlay = raw_image.copy()
    red_mask = np.zeros_like(raw_image)
    red_mask[:, :, 2] = mask_resized  # Red channel in BGR

    # Blend original image and red mask
    cv2.addWeighted(red_mask, 0.6, overlay, 0.4, 0, overlay)
    side_by_side = np.hstack([raw_image, overlay])

    cv2.imwrite(output_path, side_by_side)
    print(f"[✓] Inference successful! Visual comparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Inference for SAR Oil Spill Segmentation")
    parser.add_argument("--image", type=str, required=True, help="Path to input SAR image")
    parser.add_argument("--checkpoint", type=str, default="best_unet_model.pth", help="Path to model weights (.pth)")
    parser.add_argument("--output", type=str, default="prediction_result.png", help="Path to save visualized result")
    parser.add_argument("--img_size", type=int, default=256, help="Input resize dimension")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for mask binarization")
    args = parser.parse_args()

    predict_mask(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        img_size=args.img_size,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
