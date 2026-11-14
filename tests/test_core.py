from pathlib import Path
import numpy as np
from PIL import Image
from scipy.io import savemat
import tifffile
import torch
from torch import nn
from torch.nn import functional as F
import pytest
from dataset.segmentation import SegmentationDataset, make_loader, repeat_loader
from getnetworks import get_network
from train import training_loss, validate_splits
from utils.method import MNPAC
from utils.metrics import segmentation_scores


def test_neighborhood_matches_independent_bruteforce():
    left = torch.randn(2, 2, 4, 5, requires_grad=True)
    right = torch.randn(2, 2, 4, 5, requires_grad=True)
    actual_r, actual_l = MNPAC(left, right)

    def reference(x):
        padded = F.pad(x, (1, 1, 1, 1))
        out = torch.zeros_like(x)
        for h in range(4):
            for w in range(5):
                around = [
                    padded[:, :, h + y, w + z]
                    for y in range(3)
                    for z in range(3)
                    if (y, z) != (1, 1)
                ]
                out[:, :, h, w] = torch.stack(around).max(0).values
        return out

    beta = np.exp(-0.5)
    assert torch.allclose(actual_r, right + beta * reference(left))
    assert torch.allclose(actual_l, left + beta * reference(right))
    (actual_l.square().mean() + actual_r.square().mean()).backward()
    assert left.grad.abs().sum() > 0 and right.grad.abs().sum() > 0


@pytest.mark.parametrize("dataset", ["OSCC", "MHCS"])
def test_dataset_contract_and_unlabeled_without_masks(tmp_path, dataset):
    cube = np.arange(3 * 32 * 32, dtype=np.float32).reshape(3, 32, 32)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[5:10, 8:20] = 1
    if dataset == "OSCC":
        image_dir, mask_dir = tmp_path / "roi_tif", tmp_path / "roi_label"
        image_dir.mkdir()
        mask_dir.mkdir()
        tifffile.imwrite(image_dir / "case_0.tif", cube, photometric="minisblack")
        tifffile.imwrite(mask_dir / "case_0.tif", mask)
        name = "case_0.tif"
    else:
        image_dir, mask_dir = tmp_path / "image_256", tmp_path / "label_1"
        image_dir.mkdir()
        mask_dir.mkdir()
        savemat(image_dir / "case.mat", {"data": cube.transpose(1, 2, 0)})
        Image.fromarray(mask).save(mask_dir / "case.png")
        name = "case"
    split = tmp_path / "split.txt"
    split.write_text(name + "\n")
    data = SegmentationDataset(
        tmp_path, split, dataset, training=False, in_channels=3, image_size=32
    )
    x, y = data[0]
    assert torch.equal(x, torch.from_numpy(cube))
    assert torch.equal(y, torch.from_numpy(mask.astype(np.int64)))
    assert len(list(make_loader(data, 4))) == 1  # No dropped validation tail.
    for file in mask_dir.iterdir():
        file.unlink()
    un = SegmentationDataset(
        tmp_path, split, dataset, unsupervised=True, in_channels=3, image_size=32
    )
    assert len(un[0]) == 2
    iterator = repeat_loader(make_loader(un, 1))
    for _ in range(3):
        assert next(iterator)[0].shape == (1, 3, 32, 32)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Conv2d(3, 4, 3, padding=1)
        self.head = nn.Conv2d(4, 2, 1)

    def forward(self, x):
        f = self.feature(x).relu()
        return self.head(f), f


def test_training_loss_gradients_and_right_supervision():
    models = [TinyModel(), TinyModel()]
    images = torch.randn(2, 3, 8, 8)
    targets = torch.randint(2, (2, 8, 8))
    loss, parts = training_loss(*models, images, targets, images.flip(-1), 0.1, 1000)
    assert torch.isfinite(loss)
    # Right supervision must depend only on the right model.
    grads = torch.autograd.grad(
        parts["supervised_right"],
        list(models[0].parameters()),
        allow_unused=True,
        retain_graph=True,
    )
    assert all(g is None for g in grads)
    loss.backward()
    for model in models:
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        assert model.head.weight.grad.abs().sum() > 0


def test_metrics_empty_and_false_positive():
    truth = torch.zeros(2, 4, 4, dtype=torch.long)
    pred = truth.clone()
    pred[1, 0, 0] = 1
    assert segmentation_scores(pred, truth) == [(1.0, 1.0), (0.0, 0.0)]


def test_disjoint_custom_splits(tmp_path):
    (tmp_path / "train_labeled_4.txt").write_text("Case_1_0.tif\n")
    (tmp_path / "train_unlabeled_4.txt").write_text("Case_2_0.tif\n")
    (tmp_path / "test.txt").write_text("Case_3_0.tif\n")
    assert len(validate_splits(tmp_path, 4, "OSCC")) == 3


def test_cinet_channels_and_classes():
    # Check the full model with a reduced spatial size.
    torch.set_num_threads(2)
    model = get_network("cinet", 3, 2, 32).eval()
    with torch.no_grad():
        logits, features = model(torch.randn(1, 3, 32, 32))
    assert logits.shape == (1, 2, 32, 32)
    assert features.shape[-2:] == (32, 32)
    assert model.StructuralEnc.embeddings.hybrid_model.root.conv.in_channels == 3


def test_rejects_leaking_custom_split(tmp_path):
    (tmp_path / "train_labeled_4.txt").write_text("sample.mat\n")
    (tmp_path / "train_unlabeled_4.txt").write_text("other\n")
    (tmp_path / "test.txt").write_text("sample\n")
    with pytest.raises(ValueError, match="overlap"):
        validate_splits(tmp_path, 4, "MHCS")


def test_missing_split_has_clear_error(tmp_path):
    with pytest.raises(ValueError, match="check --split_dir"):
        validate_splits(tmp_path, 4, "MHCS")


def test_ppc_gram_normalization_and_gradients():
    from utils.losses import PPC

    left = torch.randn(2, 3, 4, 5, requires_grad=True)
    right = torch.randn(2, 3, 4, 5, requires_grad=True)
    # Compute each Gram entry independently from the module implementation.
    gram_left = torch.einsum("bchw,bdhw->bcd", left, left) / 60
    gram_right = torch.einsum("bchw,bdhw->bcd", right, right) / 60
    frobenius = (gram_left - gram_right).square().sum((1, 2)).mean()
    loss = PPC()(left, right)
    assert torch.allclose(loss * 3**2, frobenius)
    assert PPC()(left, left).item() == 0
    loss.backward()
    assert left.grad.abs().sum() > 0
    assert right.grad.abs().sum() > 0
    assert torch.isfinite(left.grad).all() and torch.isfinite(right.grad).all()


def test_neighborhood_refines_labels_without_gradients():
    from utils.method import refine_pseudo_labels

    left = torch.empty(1, 2, 3, 3)
    left[:, 0] = 0.55
    left[:, 1] = 0.45
    right = torch.empty_like(left)
    right[:, 0] = 0.01
    right[:, 1] = 0.99
    left = left.log().requires_grad_()
    right = right.log().requires_grad_()
    refined_l, refined_r = refine_pseudo_labels(left, right)
    assert left.argmax(1)[0, 1, 1].item() == 0
    assert refined_l[0, 1, 1].item() == 1
    assert refined_r[0, 1, 1].item() == 1
    assert refined_l.dtype == refined_r.dtype == torch.long
    assert not refined_l.requires_grad and not refined_r.requires_grad
    assert refined_l.grad_fn is None and refined_r.grad_fn is None
    assert left.grad is None and right.grad is None


@pytest.mark.parametrize(
    "prob_l,prob_r,label_l,label_r",
    [(0.45, 0.99, 1, 1), (0.99, 0.01, 1, 0)],
)
def test_training_uses_refined_cross_targets_and_raw_logits(
    prob_l, prob_r, label_l, label_r
):
    from utils.losses import Dice_Loss

    class FixedModel(nn.Module):
        def __init__(self, foreground_probability):
            super().__init__()
            probability = (
                torch.tensor([1 - foreground_probability, foreground_probability])
                .view(1, 2, 1, 1)
                .expand(1, 2, 3, 3)
            )
            self.logits = nn.Parameter(probability.log().clone())

        def forward(self, images):
            logits = self.logits.expand(images.shape[0], -1, -1, -1)
            return logits, logits

    left, right = FixedModel(prob_l), FixedModel(prob_r)
    images = torch.zeros(1, 3, 3, 3)
    targets = torch.zeros(1, 3, 3, dtype=torch.long)
    weight = 0.2
    _, parts = training_loss(left, right, images, targets, images, weight, 0)
    pred_l = left.logits.expand(2, -1, -1, -1)
    pred_r = right.logits.expand(2, -1, -1, -1)
    refined_l = torch.full((2, 3, 3), label_l, dtype=torch.long)
    refined_r = torch.full_like(refined_l, label_r)
    dice = Dice_Loss()
    expected = weight * (
        F.cross_entropy(pred_l, refined_r)
        + dice(pred_l, refined_r)
        + F.cross_entropy(pred_r, refined_l)
        + dice(pred_r, refined_l)
    )
    before_refinement = weight * (
        F.cross_entropy(pred_l, pred_r.detach().argmax(1))
        + dice(pred_l, pred_r.detach().argmax(1))
        + F.cross_entropy(pred_r, pred_l.detach().argmax(1))
        + dice(pred_r, pred_l.detach().argmax(1))
    )
    assert torch.allclose(parts["mnpac"], expected)
    if prob_l == 0.45:
        assert not torch.allclose(parts["mnpac"], before_refinement)
    actual_grad = torch.autograd.grad(
        parts["mnpac"], (left.logits, right.logits), retain_graph=True
    )
    expected_grad = torch.autograd.grad(expected, (left.logits, right.logits))
    for actual, wanted in zip(actual_grad, expected_grad):
        assert torch.allclose(actual, wanted)
        assert torch.isfinite(actual).all() and actual.abs().sum() > 0
