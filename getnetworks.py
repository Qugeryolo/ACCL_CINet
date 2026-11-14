import copy
from networks.CINet import CINet, CONFIGS


def get_network(network="cinet", in_channels=60, num_classes=2, image_size=256):
    if network != "cinet":
        raise ValueError("Only CINet is supported")
    if in_channels < 1 or num_classes < 2 or image_size < 32 or image_size % 16:
        raise ValueError(
            "Invalid channels/classes; image_size must be >=32 and divisible by 16"
        )
    return CINet(
        copy.deepcopy(CONFIGS["R50-ViT-B_16"]),
        image_size,
        num_classes=num_classes,
        in_channels=in_channels,
    )
