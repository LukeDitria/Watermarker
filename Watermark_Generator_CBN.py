import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from torch.nn.utils import spectral_norm
import torchvision.models as models
import Helpers as hf


class ConditionalNorm2d(nn.Module):
    def __init__(self, channels, num_features, norm_type="bn"):
        super(ConditionalNorm2d, self).__init__()
        self.channels = channels
        if norm_type == "gn":
            self.norm = nn.GroupNorm(8, channels, affine=False, eps=1e-4)
        elif norm_type == "bn":
            self.norm = nn.BatchNorm2d(channels, affine=False, eps=1e-4)
        else:
            raise ValueError("Normalisation type not recognised.")

        self.fcw = nn.Linear(num_features, channels)
        self.fcb = nn.Linear(num_features, channels)

    def forward(self, x, features):
        out = self.norm(x)
        w = self.fcw(features)
        b = self.fcb(features)

        out = w.view(-1, self.channels, 1, 1) * out + b.view(-1, self.channels, 1, 1)
        return out


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """

    def __init__(self, channel_in, channel_out, code_size, kernel_size=3, scale_factor=2):
        super(ResUp, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_in, kernel_size, 1, kernel_size // 2)
        self.bn1 = ConditionalNorm2d(channel_in, code_size)
        self.conv2 = nn.Conv2d(channel_in, channel_out, kernel_size, 1, kernel_size // 2)
        self.bn2 = ConditionalNorm2d(channel_out, code_size)

        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 1, kernel_size // 2)

        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")

        self.act_fnc = nn.ELU()

    def forward(self, input_tuple):
        x, wm_code = input_tuple

        x = self.up_nn(x)
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x), wm_code))
        x = self.conv2(x)

        return (self.act_fnc(self.bn2(x + skip, wm_code)), wm_code)


class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_in, kernel_size, 1, kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channel_in)
        self.conv2 = nn.Conv2d(channel_in, channel_out, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channel_out)

        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 1, kernel_size // 2)

        self.act_fnc = nn.ELU()

    def forward(self, x):

        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return self.act_fnc(self.bn2(x + skip))


class Generator(nn.Module):
    """
    Watermark Generator block
    """

    def __init__(self,
                 num_watermarks=128,
                 embedding_size=128,
                 img_size=64,
                 channels=3,
                 dim=64,
                 dim_mults=(8, 4, 2, 1),
                 watermark_scale=0.025,
                 blur_kernel_size=5,
                 blur_sigma=2):
        super(Generator, self).__init__()

        self.init_fm_size = img_size // (2 ** len(dim_mults))
        if self.init_fm_size < 1:
            ValueError("Invalid number of blocks")

        self.code_embedding = nn.Embedding(num_watermarks, embedding_size)

        self.fc_1 = nn.Linear(embedding_size, embedding_size)
        self.fc_2 = nn.Linear(embedding_size, embedding_size)
        self.fc_3 = nn.Linear(embedding_size, dim_mults[0] * dim * (self.init_fm_size ** 2))
        self.res_init = ResBlock(dim_mults[0] * dim, dim_mults[0] * dim)

        widths_in = list(dim_mults)
        widths_out = (list(dim_mults[1:]) + [dim_mults[-1]])

        layer_blocks = []
        for w_in, w_out in zip(widths_in, widths_out):
            layer_blocks.append(ResUp(w_in * dim, w_out * dim, code_size=embedding_size))

        self.res_blocks = nn.Sequential(*layer_blocks)

        self.conv_out = nn.Conv2d(dim_mults[-1] * dim, channels, 3, 1, 1)
        self.act_fnc = nn.ELU()
        self.watermark_scale = watermark_scale
        self.img_size = img_size
        self.dim = dim
        self.blur = hf.GaussianBlur(kernel_size=blur_kernel_size, max_sigma=blur_sigma, sample_sigma=False)

    def forward(self, x_in, wm_index, wm_scale=None):
        embedding_vec = self.act_fnc(self.code_embedding(wm_index))

        embedding_vec = self.act_fnc(self.fc_1(embedding_vec))
        embedding_vec = self.act_fnc(self.fc_2(embedding_vec))
        x = self.act_fnc(self.fc_3(embedding_vec))

        x = x.reshape(x_in.shape[0], -1, self.init_fm_size, self.init_fm_size)
        x = self.res_init(x)

        x, _ = self.res_blocks((x, embedding_vec))
        wm_out = torch.tanh(self.conv_out(x))
        wm_out = self.blur(wm_out)

        if wm_scale is None:
            wm_scale = self.watermark_scale

        img_out = (1 - wm_scale) * x_in + wm_scale * wm_out
        outputs = {"image_out": img_out, "watermark": wm_out}

        return outputs

