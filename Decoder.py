import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from torch.nn.utils import spectral_norm
import torchvision.models as models


class ResBlock(nn.Module):
    """
    Residual down sampling block for the encoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3):
        super(ResBlock, self).__init__()
        self.conv1 = (nn.Conv2d(channel_in, channel_out // 2, kernel_size, 1, kernel_size // 2))
        self.bn1 = nn.BatchNorm2d(channel_out // 2, eps=1e-4)
        self.conv2 = (nn.Conv2d(channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2))
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.conv3 = (nn.Conv2d(channel_in, channel_out, kernel_size, 1, kernel_size // 2))

        self.act_fnc = nn.ELU()

    def forward(self, x):
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return self.act_fnc(self.bn2(x + skip))


class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3):
        super(ResDown, self).__init__()
        self.conv1 = (nn.Conv2d(channel_in, channel_out // 2, kernel_size, 2, kernel_size // 2))
        self.bn1 = nn.BatchNorm2d(channel_out // 2, eps=1e-4)
        self.conv2 = (nn.Conv2d(channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2))
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.conv3 = (nn.Conv2d(channel_in, channel_out, kernel_size, 2, kernel_size // 2))

        self.act_fnc = nn.ELU()

    def forward(self, x):
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return self.act_fnc(self.bn2(x + skip))


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=2):
        super(ResUp, self).__init__()

        self.conv1 = (nn.Conv2d(channel_in, channel_in // 2, kernel_size, 1, kernel_size // 2))
        self.bn1 = nn.BatchNorm2d(channel_in // 2, eps=1e-4)
        self.conv2 = (nn.Conv2d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2))
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.conv3 = (nn.Conv2d(channel_in, channel_out, kernel_size, 1, kernel_size // 2))

        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")

        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.up_nn(x)
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return self.act_fnc(self.bn2(x + skip))


class Decoder(nn.Module):
    def __init__(self, channels, num_outputs, ch=64, blocks=(1, 2, 4, 8)):
        super(Decoder, self).__init__()
        self.conv_in = nn.Conv2d(channels, blocks[0] * ch, 3, 1, 1)

        widths_in = list(blocks)
        widths_out = list(blocks[1:]) + [blocks[-1]]

        layer_blocks = []

        for w_in, w_out in zip(widths_in, widths_out):
            layer_blocks.append(ResDown(w_in * ch, w_out * ch))

        layer_blocks.append(ResBlock(blocks[-1] * ch, blocks[-1] * ch))
        layer_blocks.append(ResBlock(blocks[-1] * ch, blocks[-1] * ch))

        self.res_blocks = nn.Sequential(*layer_blocks)

        self.ave_pool = nn.AdaptiveAvgPool2d(4)

        self.fc_watermark = nn.Linear(blocks[-1] * ch * 4 * 4, num_outputs)
        self.fc_detector = nn.Linear(blocks[-1] * ch * 4 * 4, 1)

        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.act_fnc(self.conv_in(x))
        x = self.res_blocks(x)

        x = self.ave_pool(x)
        x = x.reshape(x.shape[0], -1)

        wm_x = self.fc_watermark(x)
        det_x = self.fc_detector(x)

        output = {"decoded": wm_x, "detected": det_x}

        return output
