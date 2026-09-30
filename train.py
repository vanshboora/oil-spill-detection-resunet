import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.resunet import LargeUNet
from utils.losses import DiceBCELoss
from utils.dataset import (
    SegmentationDataset,
    DummyDataset,
    get_transforms,
)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """
    Executes one epoch of training with mixed precision and gradient scaling.
    """
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, desc="Training", leave=False)

    for images, masks in loop:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation split and computes average Loss & Dice score.
    """
    model.eval()
    running_loss = 0.0
    total_dice = 0.0
    loop = tqdm(loader, desc="Validation", leave=False)

    for images, masks in loop:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, masks)

        running_loss += loss.item()

        # Compute validation Dice Score
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        intersection = (preds * masks).sum()
        dice = (2.0 * intersection + 1e-5) / (preds.sum() + masks.sum() + 1e-5)
        total_dice += dice.item()

    mean_loss = running_loss / max(len(loader), 1)
    mean_dice = total_dice / max(len(loader), 1)
    return mean_loss, mean_dice


def main():
    parser = argparse.ArgumentParser(description="Train Residual U-Net for SAR Oil Spill Segmentation")
    parser.add_argument("--data_dir", type=str, default="data", help="Root directory of dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--img_size", type=int, default=256, help="Spatial dimension of square input image")
    parser.add_argument("--checkpoint_path", type=str, default="best_unet_model.pth", help="Path to save best weights")
    parser.add_argument("--dummy", action="store_true", help="Run quick demo training on synthetic tensors")
    parser.add_argument("--lightweight", action="store_true", help="Use lightweight feature channels for faster training")
    parser.add_argument("--workers", type=int, default=2, help="Number of DataLoader worker processes")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using compute device: {device}")

    img_size_tuple = (args.img_size, args.img_size)
    train_transform, val_transform = get_transforms(img_size_tuple)

    # 1. Dataset & DataLoader configuration
    if args.dummy or not os.path.exists(os.path.join(args.data_dir, "images")):
        if not args.dummy:
            print(f"[!] Dataset folder '{args.data_dir}' not found. Falling back to synthetic DummyDataset for demonstration.")
        print("--> Initializing DummyDataset (Synthetic mode)...")
        train_ds = DummyDataset(length=64, img_size=img_size_tuple, transform=train_transform)
        val_ds = DummyDataset(length=32, img_size=img_size_tuple, transform=val_transform)
    else:
        train_img_dir = os.path.join(args.data_dir, "images", "train")
        train_mask_dir = os.path.join(args.data_dir, "masks", "train")
        val_img_dir = os.path.join(args.data_dir, "images", "val")
        val_mask_dir = os.path.join(args.data_dir, "masks", "val")

        print(f"--> Loading real dataset from: {args.data_dir}")
        train_ds = SegmentationDataset(train_img_dir, train_mask_dir, transform=train_transform)
        val_ds = SegmentationDataset(val_img_dir, val_mask_dir, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=(len(train_ds) > args.batch_size)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda")
    )

    # 2. Model definition
    features = [32, 64, 128, 256, 512] if args.lightweight else [64, 128, 256, 512, 1024]
    model = LargeUNet(in_channels=3, out_channels=1, features=features).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--> Initialized ResUNet ({total_params / 1e6:.2f}M parameters, features={features})")

    # 3. Loss, Optimizer, Scheduler, Scaler
    criterion = DiceBCELoss(weight_bce=0.5, weight_dice=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_dice = 0.0

    print("\n" + "=" * 65)
    print("                STARTING TRAINING LOOP")
    print("=" * 65)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Checkpoint saving on best validation Dice
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                    "features": features,
                    "img_size": args.img_size,
                },
                args.checkpoint_path,
            )
            print(f"  [+] Saved new best checkpoint to '{args.checkpoint_path}' (Dice: {val_dice:.4f})")

    print("=" * 65)
    print(f"Training completed! Best Validation Dice Score: {best_val_dice:.4f}")
    print(f"Model saved to: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
