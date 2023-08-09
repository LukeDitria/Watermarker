import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from torch.nn.utils import spectral_norm
import torchvision.models as models
import math

def batch_norm_to_group_norm(layer):
    """Iterates over a whole model (or layer of a model) and replaces every batch norm 2D with a group norm

    Args:
        layer: model or one layer of a model like resnet34.layer1 or Sequential(), ...
    """
    for name, module in layer.named_modules():
        if name:
            try:
                # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, torch.nn.BatchNorm2d):
                    num_channels = sub_layer.num_features
                    # first level of current layer or model contains a batch norm --> replacing.
                    layer._modules[name] = torch.nn.GroupNorm(8, num_channels)
            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split('.')[0]
                sub_layer = getattr(layer, name)
                sub_layer = batch_norm_to_group_norm(sub_layer)
                layer.__setattr__(name=name, value=sub_layer)
    return layer


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


class DecoderResNet18Sml(nn.Module):
    def __init__(self, num_outputs):
        super(DecoderResNet18Sml, self).__init__()
        # self.resnet = batch_norm_to_group_norm(models.resnet18())
        self.resnet = models.resnet18()

        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        in_feat = self.resnet.fc.in_features

        self.resnet.fc = nn.Identity()
        self.resnet.maxpool = nn.Identity()

        self.fc_watermark = nn.Linear(in_feat, num_outputs)
        self.fc_detector = nn.Linear(in_feat, 1)

        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.resnet(x)

        wm_x = self.fc_watermark(x)
        d_x = self.fc_detector(x)

        output = {"decoded": wm_x, "detector": d_x}

        return output


class DecoderResNet34Sml(nn.Module):
    def __init__(self, num_outputs):
        super(DecoderResNet34Sml, self).__init__()
        # self.resnet = batch_norm_to_group_norm(models.resnet18())
        self.resnet = models.resnet34()

        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        in_feat = self.resnet.fc.in_features

        self.resnet.fc = nn.Identity()
        self.resnet.maxpool = nn.Identity()

        self.fc_watermark = nn.Linear(in_feat, num_outputs)
        self.fc_detector = nn.Linear(in_feat, 1)

        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.resnet(x)

        wm_x = self.fc_watermark(x)
        d_x = self.fc_detector(x)

        output = {"decoded": wm_x, "detector": d_x}

        return output


class DecoderResNet18(nn.Module):
    def __init__(self, num_outputs):
        super(DecoderResNet18, self).__init__()
        # self.resnet = batch_norm_to_group_norm(models.resnet18())
        self.resnet = models.resnet18()

        in_feat = self.resnet.fc.in_features

        self.resnet.fc = nn.Identity()

        self.fc_watermark = nn.Linear(in_feat, num_outputs)

        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.resnet(x)

        wm_x = self.fc_watermark(x)

        output = {"decoded": wm_x}

        return output


class DecoderResNet34(nn.Module):
    def __init__(self, num_outputs):
        super(DecoderResNet34, self).__init__()
        # self.resnet = batch_norm_to_group_norm(models.resnet18())
        self.resnet = models.resnet34()

        in_feat = self.resnet.fc.in_features

        self.resnet.fc = nn.Identity()

        self.fc_watermark = nn.Linear(in_feat, num_outputs)

        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.resnet(x)

        wm_x = self.fc_watermark(x)

        output = {"decoded": wm_x}

        return output



