import argparse
import json
from pathlib import Path
import torch
from dataset.segmentation import SegmentationDataset, make_loader
from getnetworks import get_network
from train import evaluate


def main():
    p = argparse.ArgumentParser(description="Evaluate a CINet checkpoint.")
    p.add_argument("--dataset", choices=["OSCC", "MHCS"], required=True)
    p.add_argument("--root_path", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split_file", type=Path, required=True)
    p.add_argument("--in_channels", type=int, default=60)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    split = a.split_file
    data = SegmentationDataset(
        a.root_path,
        split,
        a.dataset,
        training=False,
        in_channels=a.in_channels,
        image_size=a.image_size,
    )
    model = get_network("cinet", a.in_channels, 2, a.image_size).to(a.device)
    model.load_state_dict(
        torch.load(a.checkpoint, map_location=a.device, weights_only=True)
    )
    print(json.dumps(evaluate(model, make_loader(data, 1), a.device), indent=2))


if __name__ == "__main__":
    main()
