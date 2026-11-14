import math
import torch
from torch.nn import functional as F


def MNPAC(pro_l, pro_r):
    if pro_l.shape != pro_r.shape or pro_l.ndim != 4:
        raise ValueError("Expected matching BCHW tensors")

    def neighbors(x):
        padded = F.pad(x, (1, 1, 1, 1))
        height, width = x.shape[-2:]
        return torch.stack(
            [
                padded[:, :, y : y + height, z : z + width]
                for y in range(3)
                for z in range(3)
                if (y, z) != (1, 1)
            ]
        ).amax(0)

    # Return the enhanced right and left branches.
    return pro_r + math.exp(-0.5) * neighbors(pro_l), pro_l + math.exp(
        -0.5
    ) * neighbors(pro_r)


@torch.no_grad()
def refine_pseudo_labels(logits_l, logits_r):
    enhanced_r, enhanced_l = MNPAC(logits_l.softmax(1), logits_r.softmax(1))
    return enhanced_l.argmax(1), enhanced_r.argmax(1)


def sigmoid_rampup(epoch, length):
    if length == 0:
        return 1.0
    phase = 1.0 - min(max(epoch, 0.0), length) / length
    return math.exp(-5.0 * phase * phase)
