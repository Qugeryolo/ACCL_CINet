import torch


def segmentation_scores(prediction, target, num_classes=2):
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Expected matching BHW tensors")
    scores = []
    for pred, truth in zip(prediction, target):
        class_scores = []
        for cls in range(1, num_classes):
            p, t = pred == cls, truth == cls
            intersection = (p & t).sum().item()
            total, union = p.sum().item() + t.sum().item(), (p | t).sum().item()
            class_scores.append(
                (
                    2 * intersection / total if total else 1.0,
                    intersection / union if union else 1.0,
                )
            )
        scores.append(
            tuple(sum(s[i] for s in class_scores) / len(class_scores) for i in range(2))
        )
    return scores
