# 🌊 Deep Residual U-Net for SAR Oil Spill Semantic Segmentation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning semantic segmentation pipeline designed to detect marine oil spills in **Synthetic Aperture Radar (SAR)** imagery using a deep **Residual U-Net (ResUNet)** and a **Hybrid Dice-BCE Loss**.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Team Responsibilities](#-team-responsibilities)
- [Model Architecture & Loss](#-model-architecture--loss)
- [Dataset Setup](#-dataset-setup)
- [Installation](#-installation)
- [Usage & Execution](#-usage--execution)
  - [1. Sanity Checks](#1-sanity-checks)
  - [2. Quick Demo (Synthetic Tensors)](#2-quick-demo-synthetic-tensors)
  - [3. Full Training on Real Dataset](#3-full-training-on-real-dataset)
  - [4. Model Inference & Visualization](#4-model-inference--visualization)
- [References](#-references)

---

## 🔬 Overview

Marine oil spills cause devastating environmental and ecological damage. Synthetic Aperture Radar (SAR) sensors provide day-and-night, all-weather observation, but SAR images suffer from speckle noise, dark look-alikes (low-wind zones, biogenic slicks), and high class imbalance.

This project delivers:
- **Synchronized Data Pipeline**: Albumentations pipeline applying synchronized spatial transformations (flips, rotations, affine shifts) to both images and ground truth masks, alongside radiometric augmentations (color jitter, Gaussian noise).
- **Residual U-Net Architecture**: A 5-level deep encoder-decoder network featuring residual double-convolution blocks with 1x1 projection shortcuts to eliminate vanishing gradients and preserve multi-scale spatial details (~123M parameters).
- **Hybrid Objective Function**: Combined Binary Cross-Entropy (for smooth gradient propagation) and Soft Dice Loss (to combat extreme foreground-background class imbalance).
- **Mixed-Precision Training**: Integrated Automatic Mixed Precision (`torch.amp`), Cosine Annealing learning rate scheduling, and validation Dice metric tracking.

---

## 📁 Repository Structure

```text
oil-spill-detection-resunet/
│
├── models/
│   ├── __init__.py
│   └── resunet.py            # ResidualDoubleConv & LargeUNet architecture
│
├── utils/
│   ├── __init__.py
│   ├── dataset.py            # Albumentations transforms, SegmentationDataset & DummyDataset
│   └── losses.py             # Hybrid DiceBCELoss implementation
│
├── train.py                  # End-to-end training, validation & checkpointing script
├── predict.py                # Single image inference & overlay visualization
├── requirements.txt          # Python dependencies
├── .gitignore                # Excludes checkpoints, large datasets & caches
└── README.md                 # Project documentation
```

---

## 👥 Team Responsibilities

This project was built through a 3-part modular division:

| Member | Core Module | Key Responsibilities |
|---|---|---|
| **👤 Person 1** | **Data Preprocessing & Loading** (`utils/dataset.py`) | Handled SAR image and mask loading, synchronized spatial augmentations (flips, affine transformations), image-only noise injection, ImageNet normalization, mask binarization, and PyTorch `DataLoader` pipelines. |
| **👤 Person 2** | **Model Architecture & Loss** (`models/resunet.py`, `utils/losses.py`) | Designed the `ResidualDoubleConv` blocks with residual shortcut projections, implemented the 5-stage `LargeUNet` encoder-bottleneck-decoder with skip connections, and implemented the hybrid `DiceBCELoss`. |
| **👤 Person 3** | **Training & Evaluation Pipeline** (`train.py`, `predict.py`) | Developed the mixed-precision training loop (`train_one_epoch`), validation routine with Dice evaluation, Cosine Annealing learning rate scheduling, best checkpoint persistence, and inference scripts. |

---

## 🧠 Model Architecture & Loss

### Architecture Pipeline
```text
SAR Input [B, 3, 256, 256]
          │
      [Encoder]
  ResBlock (64)  ──(skip)──►  Decoder Block (64)   ──► 1x1 Conv ──► Mask [B, 1, 256, 256]
  ResBlock (128) ──(skip)──►  Decoder Block (128)
  ResBlock (256) ──(skip)──►  Decoder Block (256)
  ResBlock (512) ──(skip)──►  Decoder Block (512)
  ResBlock (1024)──(skip)──►  Decoder Block (1024)
          │
     [Bottleneck]
    ResBlock (2048)
```

### Hybrid Loss Function
$$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$$
where:
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 |X \cap Y| + \epsilon}{|X| + |Y| + \epsilon}$$

---

## 🗄 Dataset Setup

The pipeline is formatted to work with the **Refined Deep-SAR Oil Spill (SOS) Dataset** (available on [Zenodo](https://zenodo.org/)):

1. Download `images.zip` and `masks.zip`.
2. Extract and arrange them inside a `data/` directory with `train` and `val` splits:

```text
data/
├── images/
│   ├── train/      # Training SAR images (.png, .jpg, .tif)
│   └── val/        # Validation SAR images
└── masks/
    ├── train/      # Ground-truth binary masks (matching image filenames)
    └── val/        # Validation binary masks
```

> **Note**: Image and mask filenames must match in their respective folders for 1-to-1 pairing.

---

## ⚙ Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/oil-spill-detection-resunet.git
cd oil-spill-detection-resunet

# 2. (Optional) Create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage & Execution

### 1. Sanity Checks
Verify that the dataset transforms and model architecture operate correctly:
```bash
python utils/dataset.py
python models/resunet.py
```

### 2. Quick Demo (Synthetic Tensors)
You can test the entire training and validation workflow immediately without needing to download the raw dataset first:
```bash
python train.py --dummy --lightweight --epochs 5 --batch_size 4
```

### 3. Full Training on Real Dataset
To train on the downloaded dataset:
```bash
python train.py --data_dir data --epochs 25 --batch_size 8 --lr 1e-4 --checkpoint_path best_unet_model.pth
```

**Key Arguments:**
- `--data_dir`: Root dataset path containing `images/` and `masks/` (default: `data`).
- `--epochs`: Total training iterations (default: `10`).
- `--batch_size`: Batch size per step (default: `8`).
- `--lr`: Learning rate for AdamW optimizer (default: `1e-4`).
- `--img_size`: Square image resize dimension (default: `256`).
- `--lightweight`: Uses `[32, 64, 128, 256, 512]` channels for faster training on consumer GPUs.
- `--dummy`: Runs demo training using synthetic data.

### 4. Model Inference & Visualization
Generate a side-by-side segmentation prediction overlay for any SAR image:
```bash
python predict.py --image path/to/sar_image.png --checkpoint best_unet_model.pth --output prediction_result.png
```

---

## 📜 References
- **Refined Deep-SAR Oil Spill (SOS) Dataset**: Zenodo SAR Oil Spill benchmark.
- **Ronneberger et al.**: *U-Net: Convolutional Networks for Biomedical Image Segmentation* (MICCAI 2015).
- **Zhang et al.**: *Road Extraction by Deep Residual U-Net* (IEEE GRSL 2018).
