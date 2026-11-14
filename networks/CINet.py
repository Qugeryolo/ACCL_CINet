from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import math
from os.path import join as pjoin
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
import ml_collections
from collections import OrderedDict


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({"size": (16, 16)})
    config.hidden_size = 768
    config.StructuralEnc = ml_collections.ConfigDict()
    config.StructuralEnc.mlp_dim = 3072
    config.StructuralEnc.num_heads = 12
    config.StructuralEnc.num_layers = 12
    config.StructuralEnc.num_layers_p = 1
    config.StructuralEnc.attention_dropout_rate = 0.0
    config.StructuralEnc.dropout_rate = 0.1

    config.classifier = "seg"
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = "../model/vit_checkpoint/imagenet21k/ViT-B_16.npz"
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = "softmax"
    return config


def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = "seg"
    config.pretrained_path = "../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz"
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 16
    config.n_skip = 3
    config.activation = "softmax"

    return config


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(
        cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups
    )


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block."""

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or cin != cout:
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, "downsample"):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y


class ContextualEnc(nn.Module):
    def __init__(self, block_units, width_factor, in_channels=1):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        StdConv2d(
                            in_channels,
                            width,
                            kernel_size=7,
                            stride=2,
                            bias=False,
                            padding=3,
                        ),
                    ),
                    ("gn", nn.GroupNorm(32, width, eps=1e-6)),
                    ("relu", nn.ReLU(inplace=True)),
                    # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
                ]
            )
        )

        self.body = nn.Sequential(
            OrderedDict(
                [
                    (
                        "block1",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width, cout=width * 4, cmid=width
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 4, cout=width * 4, cmid=width
                                        ),
                                    )
                                    for i in range(2, block_units[0] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block2",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width * 4,
                                            cout=width * 8,
                                            cmid=width * 2,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 8,
                                            cout=width * 8,
                                            cmid=width * 2,
                                        ),
                                    )
                                    for i in range(2, block_units[1] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block3",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width * 8,
                                            cout=width * 16,
                                            cmid=width * 4,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 16,
                                            cout=width * 16,
                                            cmid=width * 4,
                                        ),
                                    )
                                    for i in range(2, block_units[2] + 1)
                                ],
                            )
                        ),
                    ),
                ]
            )
        )

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(
                    x.size(), right_size
                )
                feat = torch.zeros(
                    (b, x.size()[1], right_size, right_size), device=x.device
                )
                feat[:, :, 0 : x.size()[2], 0 : x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {
    "gelu": torch.nn.functional.gelu,
    "relu": torch.nn.functional.relu,
    "swish": swish,
}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.StructuralEnc["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.StructuralEnc["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.StructuralEnc["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.StructuralEnc["mlp_dim"])
        self.fc2 = Linear(config.StructuralEnc["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.StructuralEnc["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings."""

    def __init__(self, config, img_size, in_channels):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:  # ResNet
            grid_size = config.patches["grid"]
            patch_size = (
                img_size[0] // 16 // grid_size[0],
                img_size[1] // 16 // grid_size[1],
            )
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (
                img_size[1] // patch_size_real[1]
            )
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ContextualEnc(
                block_units=config.resnet.num_layers,
                width_factor=config.resnet.width_factor,
                in_channels=in_channels,
            )
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        # self.position_embeddings = nn.Parameter(torch.zeros(1, 256, config.hidden_size))

        self.dropout = Dropout(config.StructuralEnc["dropout_rate"])

    def forward(self, x):

        x1, features = self.hybrid_model(x)

        x = self.patch_embeddings(x1)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        # embeddings = x + self.position_embeddings
        embeddings = self.dropout(x)
        return embeddings, features, x1


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.StructuralEnc["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class StructuralEnc(nn.Module):
    def __init__(self, config, img_size, vis, in_channels=1):
        super(StructuralEnc, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size, in_channels=in_channels)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features, cnn_f = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features, cnn_f


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class EmbeddingsP(nn.Module):
    """Construct the embeddings from patch, position embeddings."""

    def __init__(
        self,
    ):
        super(EmbeddingsP, self).__init__()

        self.dropout = Dropout(0.5)

    def forward(self, x):
        # x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = self.dropout(x)
        return embeddings


class CrossIndicationAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim=512, v_dim=512, num_heads=1):
        super(CrossIndicationAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)

        self.proj_q2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v = nn.Linear(in_dim1, v_dim * num_heads, bias=False)
        self.proj_o1 = nn.Linear(v_dim * num_heads, in_dim1)
        self.proj_o2 = nn.Linear(v_dim * num_heads, in_dim2)
        # self.ffn_norm = LayerNorm(configs.hidden_size, eps=1e-6)
        # self.ffn = Mlp(configs)

    def forward(self, x1, x2, mask=None):
        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size(1)

        q1 = (
            self.proj_q1(x1)
            .view(batch_size, seq_len1, self.num_heads, self.k_dim)
            .permute(0, 2, 1, 3)
        )
        k2 = (
            self.proj_k2(x2)
            .view(batch_size, seq_len2, self.num_heads, self.k_dim)
            .permute(0, 2, 3, 1)
        )
        v2 = (
            self.proj_v(x2)
            .view(batch_size, seq_len2, self.num_heads, self.v_dim)
            .permute(0, 2, 1, 3)
        )

        attn1 = torch.matmul(q1, k2) / self.k_dim**0.5

        q2 = (
            self.proj_q2(x2)
            .view(batch_size, seq_len2, self.num_heads, self.k_dim)
            .permute(0, 2, 1, 3)
        )
        k1 = (
            self.proj_k1(x1)
            .view(batch_size, seq_len1, self.num_heads, self.k_dim)
            .permute(0, 2, 3, 1)
        )
        v1 = (
            self.proj_v(x1)
            .view(batch_size, seq_len1, self.num_heads, self.v_dim)
            .permute(0, 2, 1, 3)
        )
        attn2 = torch.matmul(q2, k1) / self.k_dim**0.5

        if mask is not None:
            attn1 = attn1.masked_fill(mask == 0, -1e9)
            attn2 = attn2.masked_fill(mask == 0, -1e9)

        attn1 = F.softmax(attn1, dim=-1)
        output1 = (
            torch.matmul(attn1, v2)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len1, -1)
        )
        output1 = self.proj_o1(output1)

        attn2 = F.softmax(attn2, dim=-1)
        output2 = (
            torch.matmul(attn2, v1)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len2, -1)
        )
        output2 = self.proj_o2(output2)

        output = torch.cat([output1, output2], dim=-1)

        return output


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels=0,
        use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        # size = [1, 512, 16, 16]
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


class EnconderEnsemble(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.embeddings = EmbeddingsP()
        self.cpa = CrossIndicationAttention(out_channels, out_channels)

    def forward(self, x, features):
        # structure StructuralEnc encoder features
        x = self.conv1(x)
        x = self.embeddings(x)

        features = self.conv1(features)
        p = self.embeddings(features)

        cpa = self.cpa(x, p)
        B, n_patch, hidden = (
            cpa.size()
        )  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = cpa.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)

        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        super().__init__(conv2d, upsampling)


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        out_channels = 1024
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(
                4 - self.config.n_skip
            ):  # re-select the skip channels according to n_skip
                skip_channels[3 - i] = 0

        else:
            skip_channels = [0, 0, 0, 0]
        # channels1 = [512, 256, 128]
        # channels2 = [512, 256, 64]
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch)
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.ensemble = EnconderEnsemble(in_channels=1024, out_channels=256)

    def forward(self, hidden_states, features=None, encoder_cnn=None):
        B, n_patch, hidden = (
            hidden_states.size()
        )  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        x = self.ensemble(x, encoder_cnn)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)

        return x


class CINet(nn.Module):
    def __init__(
        self,
        config,
        img_size,
        num_classes=2,
        zero_head=False,
        vis=False,
        in_channels=60,
    ):
        super(CINet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        config = copy.deepcopy(config)
        config.n_classes = num_classes
        config.patches.grid = (img_size // 16, img_size // 16)
        self.StructuralEnc = StructuralEnc(
            config, img_size, vis, in_channels=in_channels
        )
        self.decoder = Decoder(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config["decoder_channels"][-1],
            out_channels=config["n_classes"],
            kernel_size=3,
        )
        self.config = config

        # DTC
        self.tanh = nn.Tanh()
        dim_in = 16
        feat_dim = 32
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )
        for class_c in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1),
            )
            self.__setattr__("contrastive_class_selector_" + str(class_c), selector)

        for class_c in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1),
            )
            self.__setattr__(
                "contrastive_class_selector_memory" + str(class_c), selector
            )

    def forward(self, x):
        # if x.size()[1] == 1:
        #     x = x.repeat(1, 3, 1, 1)
        x, attn_weights, features, cnn_f = self.StructuralEnc(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features, cnn_f)
        logits = self.segmentation_head(x)
        tan = self.tanh(logits)  # DTC
        # DTC
        # return tan, logits
        return logits, x

    def forward_projection_head(self, features):
        return self.projection_head(features)

    def forward_prediction_head(self, features):
        return self.prediction_head(features)


CONFIGS = {"ViT-B_16": get_b16_config(), "R50-ViT-B_16": get_r50_b16_config()}
