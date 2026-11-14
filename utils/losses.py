import torch
from torch import nn
from torch.nn import functional as F


class Dice_Loss(nn.Module):
    def __init__(self, smooth=1.0, reduction="mean", ignore_index=-100):
        super(Dice_Loss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # softmax函数将用于输入，targets不应该被softmax处理
        inputs = F.softmax(inputs, dim=1)  # 假设inputs是[batch_size, classes, H, W]

        if self.ignore_index not in (None, -100):
            # 创建一个mask，忽略ignore_index对应的像素
            mask = (targets != self.ignore_index).float()
            # 只在非忽略的像素上计算loss
            inputs = inputs * mask.unsqueeze(1)
            targets = targets * mask

        # 选择前景类别的概率，这里假设前景是第二个通道
        inputs_foreground = inputs[:, 1, :, :]

        # 展平inputs和targets
        inputs_flat = inputs_foreground.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)

        # 计算交集
        intersection = (inputs_flat * targets_flat).sum()

        # 计算Dice系数
        dice_coef = (2.0 * intersection + self.smooth) / (
            inputs_flat.sum() + targets_flat.sum() + self.smooth
        )

        # 根据reduction参数返回loss
        if self.reduction == "mean":
            return 1 - dice_coef
        elif self.reduction == "sum":
            return (1 - dice_coef) * inputs_flat.size(0)
        elif self.reduction == "none":
            return 1 - dice_coef
        else:
            raise ValueError(f"Unsupported reduction mode: {self.reduction}")


class PPC(nn.Module):
    def __init__(self):
        super(PPC, self).__init__()

    def forward(self, input_features, target_features):
        B, C, H, W = input_features.size()
        input_features = input_features.view(B, C, -1)
        target_features = target_features.view(B, C, -1)

        input_gram = torch.bmm(input_features, input_features.transpose(1, 2)) / (
            C * H * W
        )
        target_gram = torch.bmm(target_features, target_features.transpose(1, 2)) / (
            C * H * W
        )

        loss = nn.functional.mse_loss(input_gram, target_gram)

        return loss
