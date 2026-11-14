import argparse
import json
import logging
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dataset.segmentation import SegmentationDataset, make_loader, repeat_loader
from getnetworks import get_network
from utils.losses import Dice_Loss, PPC
from utils.method import refine_pseudo_labels, sigmoid_rampup
from utils.metrics import segmentation_scores


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train ACCL-CINet on OSCC or MHCS.")
    p.add_argument("--dataset", choices=["OSCC", "MHCS"], required=True)
    p.add_argument(
        "--root_path",
        type=Path,
        required=True,
        help="Dataset directory containing image/mask subdirectories",
    )
    p.add_argument(
        "--split_dir", type=Path, required=True, help="Directory containing split lists"
    )
    p.add_argument(
        "--snapshot_dir", type=Path, help="Default: runs/<dataset>/ratio_<ratio>"
    )
    p.add_argument("--model", choices=["cinet"], default="cinet")
    p.add_argument("--in_channels", type=int, default=60)
    p.add_argument("--num_classes", type=int, choices=[2], default=2)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--nepochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--labeled_ratio", type=int, choices=[4, 8, 16, 32], default=4)
    p.add_argument(
        "--niters_per_epoch", type=int, help="Default: number of unlabeled batches"
    )
    p.add_argument("--base_lr", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.99)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--lr_power", type=float, default=0.9)
    p.add_argument("--warm_up_epoch", type=int, default=0)
    p.add_argument("--consistency", type=float, default=1.0)
    p.add_argument("--consistency_rampup", type=float, default=100.0)
    p.add_argument(
        "--ppc_weight", "--con_alpha", dest="ppc_weight", type=float, default=1000.0
    )
    p.add_argument("--eval_every", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", type=int, choices=[0, 1], default=1)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p


def initialize(model, mode):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            (nn.init.kaiming_normal_ if mode == "kaiming" else nn.init.xavier_normal_)(
                layer.weight
            )
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
    return model


def training_loss(
    model1, model2, images, targets, unsup_images, mnpac_weight, ppc_weight
):
    pred_sup_l, fea_sup_l = model1(images)
    pred_unsup_l, fea_unsup_l = model1(unsup_images)
    pred_sup_r, fea_sup_r = model2(images)
    pred_unsup_r, fea_unsup_r = model2(unsup_images)
    pred_l = torch.cat([pred_sup_l, pred_unsup_l])
    pred_r = torch.cat([pred_sup_r, pred_unsup_r])
    pseudo_l, pseudo_r = refine_pseudo_labels(pred_l, pred_r)
    ce, dice, ppc = nn.CrossEntropyLoss(), Dice_Loss(), PPC()
    loss_mnpac = (
        ce(pred_l, pseudo_r)
        + dice(pred_l, pseudo_r)
        + ce(pred_r, pseudo_l)
        + dice(pred_r, pseudo_l)
    ) * mnpac_weight
    loss_ppc = (ppc(fea_sup_l, fea_sup_r) + ppc(fea_unsup_l, fea_unsup_r)) * ppc_weight
    sup_l = ce(pred_sup_l, targets) + dice(pred_sup_l, targets)
    sup_r = ce(pred_sup_r, targets) + dice(pred_sup_r, targets)
    return sup_l + sup_r + loss_mnpac + loss_ppc, {
        "supervised_left": sup_l,
        "supervised_right": sup_r,
        "mnpac": loss_mnpac,
        "ppc": loss_ppc,
    }


@torch.no_grad()
def evaluate(model, loader, device, num_classes=2):
    model.eval()
    scores = []
    for images, labels in loader:
        outputs, _ = model(images.to(device))
        outputs = F.interpolate(
            outputs, labels.shape[-2:], mode="bilinear", align_corners=True
        )
        scores.extend(segmentation_scores(outputs.argmax(1).cpu(), labels, num_classes))
    if not scores:
        raise ValueError("Evaluation split is empty")
    return {
        "dice": float(np.mean([s[0] for s in scores])),
        "iou": float(np.mean([s[1] for s in scores])),
        "samples": len(scores),
    }


def validate_splits(split_dir, ratio, dataset):
    if not (split_dir / f"train_labeled_{ratio}.txt").is_file():
        raise ValueError(
            f"Split ratio {ratio} unavailable in {split_dir}; check --split_dir"
        )
    paths = [
        split_dir / f"train_labeled_{ratio}.txt",
        split_dir / f"train_unlabeled_{ratio}.txt",
        split_dir / "test.txt",
    ]
    groups = [
        set(x.strip() for x in p.read_text().splitlines() if x.strip()) for p in paths
    ]
    # Compare case IDs for OSCC, which may also be supplied as patch filenames.
    if dataset == "OSCC":
        groups = [
            {Path(n).stem.rsplit("_", 1)[0] if Path(n).suffix else n for n in g}
            for g in groups
        ]
    else:
        groups = [{Path(n).stem for n in g} for g in groups]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        overlap = groups[a] & groups[b]
        if overlap:
            raise ValueError(
                f"Split overlap between {paths[a].name} and {paths[b].name}: {sorted(overlap)[:3]}"
            )
    return paths


def validate_args(args: argparse.Namespace) -> None:
    """Validate training settings before allocating models."""
    if args.nepochs < 1 or args.eval_every < 1 or args.base_lr <= 0:
        raise ValueError("nepochs/eval_every/base_lr must be positive")
    if args.consistency_rampup < 0 or not 0 <= args.warm_up_epoch < args.nepochs:
        raise ValueError("Invalid rampup/warmup duration")
    if args.niters_per_epoch is not None and args.niters_per_epoch < 1:
        raise ValueError("niters_per_epoch must be positive")
    if args.in_channels < 1 or args.image_size < 32 or args.image_size % 16:
        raise ValueError(
            "in_channels must be positive; image_size must be >=32 and divisible by 16"
        )


def seed_everything(seed: int, deterministic: bool) -> None:
    """Seed the training process and configure cuDNN behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def build_loaders(args: argparse.Namespace):
    """Build labeled, unlabeled, and evaluation loaders in that order."""
    split_dir = args.split_dir
    split_paths = validate_splits(split_dir, args.labeled_ratio, args.dataset)
    common = dict(
        root=args.root_path,
        dataset=args.dataset,
        in_channels=args.in_channels,
        image_size=args.image_size,
        num_classes=args.num_classes,
    )
    datasets = [
        SegmentationDataset(
            split_file=p, training=i != 2, unsupervised=i == 1, **common
        )
        for i, p in enumerate(split_paths)
    ]
    loaders = [
        make_loader(
            d, args.batch_size if i != 2 else 1, args.num_workers, args.seed + i
        )
        for i, d in enumerate(datasets)
    ]
    return loaders


def configure_run(args: argparse.Namespace) -> Path:
    """Create the run directory, persist settings, and configure logging."""
    out = (
        args.snapshot_dir or Path("runs") / args.dataset / f"ratio_{args.labeled_ratio}"
    )
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists():
        raise FileExistsError(
            f"{out} already contains a run; choose a new --snapshot_dir"
        )
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out / "train.log")],
        force=True,
    )
    return out


def build_models(args: argparse.Namespace, device: torch.device):
    """Initialize two independent CINet branches and their optimizers."""
    models = [
        initialize(
            get_network(
                args.model, args.in_channels, args.num_classes, args.image_size
            ),
            mode,
        ).to(device)
        for mode in ("kaiming", "xavier")
    ]
    optimizers = [
        torch.optim.SGD(
            m.parameters(),
            lr=args.base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        for m in models
    ]
    return models, optimizers


def train(args: argparse.Namespace) -> Path:
    """Train both branches and evaluate them at the configured interval."""
    validate_args(args)
    seed_everything(args.seed, bool(args.deterministic))
    device = torch.device(args.device)
    loaders = build_loaders(args)
    args.niters_per_epoch = args.niters_per_epoch or len(loaders[1])
    out = configure_run(args)
    models, optimizers = build_models(args, device)
    best = [-1.0, -1.0]
    total_steps = args.nepochs * args.niters_per_epoch
    warmup_steps = args.warm_up_epoch * args.niters_per_epoch
    for epoch in range(args.nepochs):
        for model in models:
            model.train()
        labeled, unlabeled = repeat_loader(loaders[0]), repeat_loader(loaders[1])
        weight = args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)
        epoch_loss = 0.0
        for step in range(args.niters_per_epoch):
            current = epoch * args.niters_per_epoch + step
            lr = args.base_lr * (1 - current / total_steps) ** args.lr_power
            if current < warmup_steps:
                lr = args.base_lr * current / warmup_steps
            for opt in optimizers:
                for group in opt.param_groups:
                    group["lr"] = lr
                opt.zero_grad(set_to_none=True)
            images, targets = next(labeled)
            unsup_images, _ = next(unlabeled)
            loss, _ = training_loss(
                *models,
                images.to(device),
                targets.to(device),
                unsup_images.to(device),
                weight,
                args.ppc_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch+1}, step {step+1}"
                )
            loss.backward()
            for opt in optimizers:
                opt.step()
            epoch_loss += loss.item()
        record = {
            "epoch": epoch + 1,
            "loss": epoch_loss / args.niters_per_epoch,
            "lr": lr,
            "mnpac_weight": weight,
        }
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.nepochs:
            for i, model in enumerate(models):
                result = evaluate(model, loaders[2], device, args.num_classes)
                record[f"model_{i+1}"] = result
                torch.save(model.state_dict(), out / f"last_model_{i+1}.pth")
                if result["dice"] > best[i]:
                    best[i] = result["dice"]
                    torch.save(model.state_dict(), out / f"best_model_{i+1}.pth")
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        logging.info(json.dumps(record))
    return out


def main() -> None:
    """Run the unified OSCC/MHCS training command."""
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
