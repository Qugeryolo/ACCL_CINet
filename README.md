# ACCL-CINet

PyTorch implementation of **ACCL-CINet** for semi-supervised medical hyperspectral image segmentation, with OSCC and MHCS data interfaces.

> **Semi-Supervised Medical Hyperspectral Image Segmentation Using Adversarial Consistency Constraint Learning and Cross Indication Network**
>
> Geng Qin, Huan Liu, Xueyu Zhang, Wei Li, Yuxing Guo, and Chuanbin Guo
>
> *IEEE Transactions on Image Processing*, vol. 34, pp. 5414–5428, 2025.
>
> [Paper](https://doi.org/10.1109/TIP.2025.3598499) · [Publication record](https://pubmed.ncbi.nlm.nih.gov/40828725/)

ACCL refers to adversarial consistency constraint learning; CINet refers to the cross indication network. Training uses a single entry point, `train.py`. The network is defined in `networks/CINet.py`.

This release has passed code and short-run checks. Full training and paper-level reproduction have not been completed. The paper's metadata and high-level description have been checked against public sources; equation-level correspondence requires the full text.

## Method

| Module | Name | Implementation |
| --- | --- | --- |
| PPC | Pixel Perceptual Consistency | `PPC` in `utils/losses.py` |
| MNPAC | Maximum Neighborhood Probability Adversarial Constraint | `MNPAC` in `utils/method.py` |

PPC compares spatial-spectral statistics of the two pixel-decoder feature maps. Each feature map is reshaped to `[B, C, H*W]`, and its Gram matrix is computed as `G = F @ F.transpose(1, 2) / (C*H*W)`. PPC is applied to both labeled and unlabeled inputs.

The implementation retains the existing mean squared difference between Gram matrices: `mean((G1 - G2)**2)`. This equals the batch-averaged squared Frobenius distance divided by `C**2`; it is not the unaveraged expression in Eq. (8). The loss scale is preserved, with `--ppc_weight` defaulting to 1000 (`--con_alpha` is accepted as an alias).

MNPAC computes per-class maxima over the eight adjacent pixels, excluding the center. Each branch's probabilities are enhanced using the other branch's neighborhood maximum, with a factor of `exp(-0.5)`. `refine_pseudo_labels` takes the argmax of these enhanced scores to generate spatially refined pseudo-labels under `torch.no_grad()`.

Training uses the refined right pseudo-labels to supervise the left logits, and the refined left pseudo-labels to supervise the right logits, with CE and Dice losses. Both labeled and unlabeled predictions participate in this cross-supervision term. Gradients flow through the supervised logits, not through pseudo-label generation. PPC and ground-truth supervision are unchanged.

```text
Left/right logits → softmax → MNPAC neighborhood enhancement
                                    ↓
                         refined pseudo-labels (no grad)
                                    ↓
               opposite branch logits → CE + Dice
```

This training path implements neighborhood refinement before cross supervision. It changes the earlier workflow that generated pseudo-labels before neighborhood enhancement; complete equation-level equivalence to the published method has not been established.

## Installation

Dependencies in `requirements.txt` match the project's `semi` environment:

| Dependency | Version |
| --- | --- |
| Python | 3.8.0 |
| PyTorch | 1.13.0+cu117 |
| NumPy | 1.24.4 |
| SciPy | 1.10.1 |
| Pillow | 10.3.0 |
| tifffile | 2023.7.10 |
| ml-collections | 0.1.1 |

Activate the existing environment:

```bash
conda activate semi
python -m pip install -r requirements.txt
```

To create a separate environment on another machine:

```bash
conda create -n accl-cinet python=3.8
conda activate accl-cinet
python -m pip install -r requirements.txt
```

The requirements include the PyTorch CUDA 11.7 wheel index. Use a compatible NVIDIA driver for GPU execution. No pretrained weights are downloaded automatically. Development tools are listed separately in `requirements-dev.txt`; Black was not installed in the inspected `semi` environment.

## Data preparation

`--root_path` points to one dataset directory:

```text
OSCC/
├── roi_tif/
│   └── Case_54_0.tif
└── roi_label/
    └── Case_54_0.tif

MHCS/
├── image_256/
│   └── sample.mat
└── label_1/
    └── sample.png
```

| Dataset | Image | Mask | Split entry |
| --- | --- | --- | --- |
| OSCC | TIFF, `[C, H, W]` | Matching TIFF, `[H, W]` | Case ID or full patch filename |
| MHCS | MATLAB `data`, `[H, W, C]` | Matching PNG, `[H, W]` | Sample stem or `.mat` filename |

Both interfaces default to 60 input channels and binary labels `0/1`. For OSCC, a case ID such as `Case_54` expands to patches `Case_54_0.tif` through `Case_54_98.tif`. Full patch filenames are also accepted. Unlabeled samples require images only.

Images retain their original intensity scale. Inputs are resized to 256 × 256 with bilinear interpolation; masks use nearest-neighbor interpolation. Training uses random vertical flips and 90° rotations. Evaluation uses no random augmentation. Convert `0/255` masks to class IDs explicitly before training.

### Splits

Split lists are supplied separately and are not included in this repository. Pass their directory with the required `--split_dir` argument. For `--labeled_ratio 4`, provide:

```text
/path/to/splits/
├── train_labeled_4.txt
├── train_unlabeled_4.txt
└── test.txt
```

The ratio selects the list suffix; it does not generate a new split. Each file contains one sample ID or filename per line. Training rejects overlapping IDs. OSCC checks compare case IDs; MHCS checks compare patch IDs. Verify patient-level independence separately.

## Training

All training uses `train.py`. Select the dataset with `--dataset`:

```bash
python train.py \
  --dataset OSCC \
  --root_path /path/to/OSCC \
  --split_dir /path/to/oscc_splits \
  --labeled_ratio 4 \
  --snapshot_dir runs/oscc_4 \
  --device cuda:0
```

```bash
python train.py \
  --dataset MHCS \
  --root_path /path/to/MHCS \
  --split_dir /path/to/mhcs_splits \
  --labeled_ratio 4 \
  --snapshot_dir runs/mhcs_4 \
  --device cuda:0
```

Run `python train.py --help` for all options. Use `--device cpu` for CPU execution.

| Setting | Default |
| --- | --- |
| Network | CINet (`--model cinet`) |
| Input / output channels | 60 / 2 |
| Image size | 256 × 256 |
| Epochs / batch size | 60 / 4 |
| Optimizer | SGD, momentum 0.99, weight decay 5e-4 |
| Initial learning rate | 1e-4 |
| PPC coefficient (`--ppc_weight`) | 1000 |
| Iterations per epoch | Number of unlabeled batches |
| Evaluation interval | Every 3 epochs and the final epoch |

Set `--niters_per_epoch` explicitly to use a fixed number of steps per epoch.

Each run stores `config.json`, `train.log`, `metrics.jsonl`, and best/last weights for both branches. Best weights are selected by evaluation Dice; last weights are updated at evaluation intervals. Weights contain model state dictionaries only. Optimizer resume is not implemented. Existing run directories are protected from overwrite.

The file named `test.txt` is used for periodic evaluation and model selection. For final experiments, provide a validation list under that name and evaluate the selected checkpoint on a separate held-out test list.

## Evaluation

```bash
python predict.py \
  --dataset OSCC \
  --root_path /path/to/OSCC \
  --checkpoint runs/oscc_4/best_model_1.pth \
  --split_file /path/to/held_out_test.txt \
  --device cuda:0
```

Match `--in_channels` and `--image_size` to training. `--split_file` is required and points to your evaluation list.

Evaluation reports foreground Dice and IoU averaged per image, plus the sample count. Two empty masks score 1; exactly one empty mask scores 0. HD and HD95 are not implemented in this release.

## Development

```bash
python -m pip install -r requirements-dev.txt
python -m black --check .
python -m pytest -q
python -m compileall -q .
```

Formatting uses Black with an 88-character target, four-space indentation, and LF line endings. Training setup, data loading, model construction, loss computation, and evaluation have separate functions.

## Citation

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
