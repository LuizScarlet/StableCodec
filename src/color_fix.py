'''
# --------------------------------------------------------------------------------
#   (Modified) Color fixed script from Li Yi (https://github.com/pkuliyi2015/sd-webui-stablesr/blob/master/srmodule/colorfix.py)
# --------------------------------------------------------------------------------
'''

import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from torchvision.transforms import ToTensor, ToPILImage


def adain_color_fix_quant(target: Image, source: Image, bits: int):
    # Convert images to tensors
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    # Apply adaptive instance normalization
    result_tensor = adaptive_instance_normalization_quant(target_tensor, source_tensor, bits)

    # Convert tensor back to image
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image


def calc_mean_std_quant(feat: Tensor, eps=1e-5, bits=10):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.reshape(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(b, c, 1, 1)
    feat_mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)

    quant_bin = 2 ** bits - 1
    feat_mean = torch.round(feat_mean * quant_bin) / quant_bin
    feat_std = torch.round(feat_std * quant_bin) / quant_bin

    return feat_mean, feat_std


def adaptive_instance_normalization_quant(content_feat:Tensor, style_feat:Tensor, bits: int):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std_quant(style_feat, bits=bits)
    content_mean, content_std = calc_mean_std_quant(content_feat, bits=bits)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)
