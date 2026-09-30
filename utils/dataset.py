import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(img_size: tuple[int, int] = (256, 256)):
    """
    Returns training and validation transform pipelines using Albumentations.
    Guarantees spatial transforms (flips, rotations, resizes) are applied
    identically to both the SAR image and the ground-truth mask.
    """
    train_transform = A.Compose([
        # Spatial Transforms (applied identically to both image and mask)
        A.Resize(height=img_size[0], width=img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),

        # Color & Noise Augmentations (applied ONLY to image)
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),

        # Normalization (ImageNet stats) & Conversion to PyTorch Tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

    return train_transform, val_transform


class SegmentationDataset(Dataset):
    """
    PyTorch Dataset loading SAR image and binary segmentation mask pairs.
    Matches image and mask filenames alphabetically inside their respective folders.
    """
    def __init__(self, image_dir: str, mask_dir: str, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            raise FileNotFoundError(
                f"Image directory '{image_dir}' or mask directory '{mask_dir}' does not exist."
            )

        # Valid image extensions
        valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(valid_exts)])

        assert len(self.images) == len(self.masks), (
            f"Number of images ({len(self.images)}) does not match masks ({len(self.masks)})"
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # Read image (BGR -> RGB)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to read image at: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read mask as grayscale [0, 255]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask at: {mask_path}")

        # Apply synchronized transformations
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Ensure tensor format: binarize to 0.0 or 1.0, shape [1, H, W]
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = (mask > 127).unsqueeze(0).to(torch.float32)

        return image, mask


class DummyDataset(Dataset):
    """
    Synthetic dataset generating random SAR-like tensors for testing
    pipelines and models without downloading the full Zenodo dataset.
    """
    def __init__(self, length: int = 64, img_size: tuple[int, int] = (256, 256), transform=None):
        self.length = length
        self.img_size = img_size
        self.transform = transform

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = np.random.randint(0, 256, (self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        mask = np.random.choice([0, 255], size=self.img_size).astype(np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img_tensor = augmented["image"]
            mask_tensor = augmented["mask"]
        else:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(mask)

        if isinstance(mask_tensor, np.ndarray):
            mask_tensor = torch.from_numpy(mask_tensor)
        mask_tensor = (mask_tensor > 127).unsqueeze(0).to(torch.float32)

        return img_tensor, mask_tensor


def get_dataloaders(
    train_img_dir: str,
    train_mask_dir: str,
    val_img_dir: str,
    val_mask_dir: str,
    batch_size: int = 8,
    img_size: tuple[int, int] = (256, 256),
    num_workers: int = 2
) -> tuple[DataLoader, DataLoader]:
    """
    Constructs PyTorch DataLoader instances for training and validation splits.
    """
    train_transform, val_transform = get_transforms(img_size=img_size)

    train_ds = SegmentationDataset(train_img_dir, train_mask_dir, transform=train_transform)
    val_ds = SegmentationDataset(val_img_dir, val_mask_dir, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test preprocessing pipeline with mock data
    train_transform, _ = get_transforms(img_size=(256, 256))
    dummy_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    dummy_mask = np.random.choice([0, 255], size=(256, 256)).astype(np.uint8)

    augmented = train_transform(image=dummy_img, mask=dummy_mask)
    img_tensor = augmented["image"]
    mask_tensor = (augmented["mask"] > 127).unsqueeze(0).float()

    print(f"Processed Image Tensor Shape: {img_tensor.shape} | Dtype: {img_tensor.dtype}")
    print(f"Processed Mask Tensor Shape:  {mask_tensor.shape} | Dtype: {mask_tensor.dtype}")
    print(f"Mask Unique Values: {torch.unique(mask_tensor).tolist()}")
    print("Preprocessing verification passed!")
