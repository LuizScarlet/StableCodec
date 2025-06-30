import math
import json
import torch
import torch.nn as nn
from .base import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import AttentionBlock, conv3x3
from .ckbd import *


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)


def model_config():
    config = Config({
        "N": 192,
        "M": 320,
        "slice_num": 10,
        "context_window": 5,
        "slice_ch": [8, 8, 8, 8, 16, 16, 32, 32, 96, 96],
    })

    return config


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


class ResidualBottleneck(nn.Module):
    def __init__(self, N=192, act=nn.ReLU) -> None:
        super().__init__()
        self.branch = nn.Sequential(
            conv1x1(N, N // 2),
            act(),
            nn.Conv2d(N // 2, N // 2, kernel_size=3, stride=1, padding=1),
            act(),
            conv1x1(N // 2, N)
        )

    def forward(self, x):
        out = x + self.branch(x)
        return out
    

class AnalysisTransformEX(nn.Module):
    def __init__(self, N, M, act=nn.ReLU):
        super().__init__()
        self.analysis_transform = nn.Sequential(
            conv(3, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            conv(N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            AttentionBlock(N),
            conv(N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            conv(N, M),
            AttentionBlock(M),
        )

    def forward(self, x):
        x = self.analysis_transform(x)
        return x
        

class HyperAnalysisEX(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.reduction = nn.Sequential(
            conv3x3(M, N),
            act(),
            conv(N, N),
            act(),
            conv(N, N)
        )

    def forward(self, x):
        x = self.reduction(x)
        return x


class SynthesisTransformEX(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            AttentionBlock(N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            deconv(N, 3),
        )

    def forward(self, x):
        x = self.synthesis_transform(x)
        return x
    

class HyperSynthesisEX(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.increase = nn.Sequential(
            deconv(N, M),
            act(),
            deconv(M, M * 3 // 2),
            act(),
            deconv(M * 3 // 2, M * 2, kernel_size=3, stride=1),
        )

    def forward(self, x):
        x = self.increase(x)
        return x


class EntropyParametersEX(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.GELU) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim, out_dim * 5 // 3, 1),
            act(),
            nn.Conv2d(out_dim * 5 // 3, out_dim * 4 // 3, 1),
            act(),
            nn.Conv2d(out_dim * 4 // 3, out_dim, 1),
        )

    def forward(self, params):
        """
        Args:
            params(Tensor): [B, C * K, H, W]
        return:
            gaussian_params(Tensor): [B, C * 2, H, W]
        """
        gaussian_params = self.fusion(params)

        return gaussian_params


class ChannelContextEX(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.ReLU) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim, 224, kernel_size=5, stride=1, padding=2),
            act(),
            nn.Conv2d(224, 128, kernel_size=5, stride=1, padding=2),
            act(),
            nn.Conv2d(128, out_dim, kernel_size=5, stride=1, padding=2)
        )

    def forward(self, channel_params):
        """
        Args:
            channel_params(Tensor): [B, C * K, H, W]
        return:
            channel_params(Tensor): [B, C * 2, H, W]
        """
        channel_params = self.fusion(channel_params)

        return channel_params


class LocalContextEX(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.fusion = nn.Conv2d(in_dim, out_dim, kernel_size=5, stride=1, padding=2)

    def forward(self, local_params):
        local_params = self.fusion(local_params)
        return local_params


class ELIC(CompressionModel):
    def __init__(self, N=192, M=320):
        super().__init__()

        config = model_config()

        N = config.N
        M = config.M
        slice_num = config.slice_num
        slice_ch = config.slice_ch
        self.slice_num = slice_num
        self.slice_ch = slice_ch
        self.g_a = AnalysisTransformEX(N, M, act=nn.ReLU)
        self.g_s = SynthesisTransformEX(N, M, act=nn.ReLU)
        self.h_a = HyperAnalysisEX(N, M, act=nn.ReLU)
        self.h_s = HyperSynthesisEX(N, M, act=nn.ReLU)
        self.local_context = nn.ModuleList(
            LocalContextEX(in_dim=slice_ch[i], out_dim=slice_ch[i] * 2)
            for i in range(len(slice_ch))
        )
        self.channel_context = nn.ModuleList(
            ChannelContextEX(in_dim=sum(slice_ch[:i]), out_dim=slice_ch[i] * 2, act=nn.ReLU) if i else None
            for i in range(slice_num)
        )
        self.entropy_parameters_anchor = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            if i else EntropyParametersEX(in_dim=M * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            for i in range(slice_num)
        )
        self.entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 4, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            if i else  EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            for i in range(slice_num)
        )

        # Gussian Conditional
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self):
        pass
