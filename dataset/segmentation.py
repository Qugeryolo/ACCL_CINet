from pathlib import Path
import random
import numpy as np
import tifffile
from PIL import Image
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F


class SegmentationDataset(Dataset):
    def __init__(
        self,
        root,
        split_file,
        dataset,
        training=True,
        unsupervised=False,
        in_channels=60,
        image_size=256,
        num_classes=2,
    ):
        self.root = Path(root)
        self.dataset = dataset.upper()
        if self.dataset not in ("OSCC", "MHCS"):
            raise ValueError("Only OSCC and MHCS are supported")
        self.training = training
        self.unsupervised = unsupervised
        self.in_channels = in_channels
        self.image_size = image_size
        self.num_classes = num_classes
        names = [
            s.strip() for s in Path(split_file).read_text().splitlines() if s.strip()
        ]
        if not names:
            raise ValueError(f"Empty split: {split_file}")
        self.samples = []
        for name in names:
            if Path(name).name != name or name in (".", ".."):
                raise ValueError(f"Split entries must be filenames or IDs: {name}")
            if self.dataset == "OSCC":
                # OSCC case IDs expand to 99 patches.
                filenames = (
                    [name]
                    if Path(name).suffix
                    else [f"{name}_{i}.tif" for i in range(99)]
                )
                self.samples.extend(
                    (self.root / "roi_tif" / n, self.root / "roi_label" / n)
                    for n in filenames
                )
            else:
                stem = Path(name).stem
                self.samples.append(
                    (
                        self.root / "image_256" / f"{stem}.mat",
                        self.root / "label_1" / f"{stem}.png",
                    )
                )
        missing = [
            str(p)
            for image, mask in self.samples
            for p in ((image,) if unsupervised else (image, mask))
            if not p.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} missing files; first: {missing[0]}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        if self.dataset == "MHCS":
            mat = loadmat(image_path)
            if "data" not in mat:
                raise KeyError(f'{image_path}: expected MATLAB key "data"')
            cube = np.asarray(mat["data"]).transpose(2, 0, 1)
        else:
            cube = np.asarray(tifffile.imread(image_path))
        if cube.ndim != 3 or cube.shape[0] != self.in_channels:
            raise ValueError(
                f"{image_path}: expected CHW with {self.in_channels} channels, got {cube.shape}"
            )
        if not np.isfinite(cube).all():
            raise ValueError(f"{image_path}: non-finite input")
        image = torch.from_numpy(np.array(cube, dtype=np.float32, copy=True))
        if self.unsupervised:
            mask = torch.zeros(image.shape[1:], dtype=torch.long)
        else:
            raw = np.asarray(
                tifffile.imread(mask_path)
                if self.dataset == "OSCC"
                else Image.open(mask_path)
            )
            if raw.ndim != 2 or raw.shape != cube.shape[1:]:
                raise ValueError(
                    f"{mask_path}: mask must match image spatial dimensions"
                )
            if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
                raise ValueError(f"{mask_path}: mask must contain integer class IDs")
            if raw.min() < 0 or raw.max() >= self.num_classes:
                raise ValueError(
                    f"{mask_path}: expected labels 0..{self.num_classes-1}; got {np.unique(raw)}"
                )
            mask = torch.from_numpy(np.array(raw, dtype=np.int64, copy=True))
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(
                image[None],
                (self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )[0]
            mask = F.interpolate(
                mask[None, None].float(),
                (self.image_size, self.image_size),
                mode="nearest",
            )[0, 0].long()
        if self.training:
            if random.random() < 0.5:
                image, mask = image.flip(-2), mask.flip(-2)
            k = random.randrange(4) if random.random() < 0.5 else 0
            image, mask = image.rot90(k, (-2, -1)), mask.rot90(k, (-2, -1))
        return image.contiguous(), mask.contiguous()


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, batch_size, num_workers=0, seed=42):
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive; num_workers must be nonnegative")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataset.training,
        drop_last=dataset.training,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
    if len(loader) == 0:
        raise ValueError("Training split is smaller than batch_size")
    return loader


def repeat_loader(loader):
    """Restart the loader without caching batches or freezing augmentation."""
    if len(loader) == 0:
        raise ValueError("Cannot repeat an empty loader")
    while True:
        yield from loader
