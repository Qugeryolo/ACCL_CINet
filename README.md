<div align="center">

# ACCL-CINet

**Semi-Supervised Medical Hyperspectral Image Segmentation**

*IEEE Transactions on Image Processing, 2025*

[![Paper](https://img.shields.io/badge/IEEE_TIP-2025-00629B?style=flat-square)](https://doi.org/10.1109/TIP.2025.3598499) [![Python](https://img.shields.io/badge/Python-3.8-3776AB?style=flat-square&logo=python&logoColor=white)](#-installation) [![PyTorch](https://img.shields.io/badge/PyTorch-1.13.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](#-installation)

[📄 Paper](https://doi.org/10.1109/TIP.2025.3598499) · [🚀 Usage](#-usage) · [📚 Citation](#-citation)

</div>

## 🔬 Introduction

ACCL-CINet combines a **Cross Indication Network (CINet)** with **Adversarial Consistency Constraint Learning (ACCL)** for semi-supervised medical hyperspectral image segmentation. It uses **Pixel Perceptual Consistency (PPC)** to align feature statistics and **Maximum Neighborhood Probability Adversarial Constraint (MNPAC)** to refine pseudo-labels for cross supervision.

**Supported datasets:** OSCC and MHCS.

| Component | Role |
| :--- | :--- |
| 🧠 **CINet** | Cross indication network for segmentation |
| 🔗 **PPC** | Feature-statistics alignment through pixel perceptual consistency |
| 🎯 **MNPAC** | Neighborhood-based pseudo-label refinement for cross supervision |

## 🗂️ Project Structure

```text
ACCL_CINet/
├── train.py                 # Training entry point
├── predict.py               # Evaluation entry point
├── getnetworks.py           # Model construction
├── networks/
│   └── CINet.py             # CINet architecture
├── dataset/
│   └── segmentation.py      # OSCC and MHCS interfaces
├── utils/
│   ├── losses.py            # PPC and Dice losses
│   ├── method.py            # MNPAC and pseudo-label refinement
│   └── metrics.py           # Dice and IoU metrics
├── tests/
│   └── test_core.py
└── requirements.txt         # Runtime dependencies
```

## ⚙️ Installation

The environment uses **Python 3.8**, **PyTorch 1.13.0+cu117**, and **CUDA 11.7**. `requirements.txt` contains the full Python package list exported from the `semi` environment, including MedPy.

```bash
conda create -n semi python=3.8 -y
conda activate semi
python -m pip install -r requirements.txt
```

If `semi` already exists, start with `conda activate semi`. Run all commands from the project directory.

## 🚀 Usage

### Prepare Data

| Dataset | Image directory | Mask directory |
| --- | --- | --- |
| OSCC | `OSCC/roi_tif/*.tif` | `OSCC/roi_label/*.tif` |
| MHCS | `MHCS/image_256/*.mat` | `MHCS/label_1/*.png` |

OSCC images use `[C, H, W]`; MHCS images use MATLAB key `data` with shape `[H, W, C]`. Both interfaces default to **60 channels** and **0/1 mask labels**. Unlabeled samples require images only.

Provide split lists separately. For `--labeled_ratio 4`, the split directory contains:

```text
splits/
├── train_labeled_4.txt
├── train_unlabeled_4.txt
└── test.txt
```

Each line contains a sample ID or filename. OSCC accepts full patch filenames or case IDs, which expand to patches `_0.tif` through `_98.tif`. MHCS accepts sample stems or `.mat` filenames. `test.txt` provides the evaluation split for periodic evaluation and checkpoint selection.

### Train

**OSCC**

```bash
python train.py \
  --dataset OSCC \
  --root_path /path/to/OSCC \
  --split_dir /path/to/oscc_splits \
  --labeled_ratio 4 \
  --snapshot_dir runs/oscc_4 \
  --device cuda:0
```

**MHCS**

```bash
python train.py \
  --dataset MHCS \
  --root_path /path/to/MHCS \
  --split_dir /path/to/mhcs_splits \
  --labeled_ratio 4 \
  --snapshot_dir runs/mhcs_4 \
  --device cuda:0
```

### Evaluate

```bash
python predict.py \
  --dataset OSCC \
  --root_path /path/to/OSCC \
  --checkpoint runs/oscc_4/best_model_1.pth \
  --split_file /path/to/held_out_test.txt \
  --device cuda:0
```

For MHCS, change `--dataset` and the corresponding paths. Evaluation reports foreground Dice and IoU. Use `python train.py --help` or `python predict.py --help` for all options.

## 📚 Citation

If you use this work, please cite:

```bibtex
@article{qin2025acclcinet,
  author  = {Qin, Geng and Liu, Huan and Zhang, Xueyu and Li, Wei and Guo, Yuxing and Guo, Chuanbin},
  title   = {Semi-Supervised Medical Hyperspectral Image Segmentation Using Adversarial Consistency Constraint Learning and Cross Indication Network},
  journal = {IEEE Transactions on Image Processing},
  year    = {2025},
  volume  = {34},
  pages   = {5414--5428},
  doi     = {10.1109/TIP.2025.3598499}
}
```
