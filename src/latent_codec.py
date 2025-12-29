import math
from torch import Tensor
from typing import NamedTuple

import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ops import quantize_ste as ste_round
from compressai.ans import BufferedRansEncoder, RansDecoder
import sys
sys.path.append("..")
from ELIC.model.elic_official import CompressionModel, get_scale_table

class InceptionDWConv2d(nn.Module):
    def __init__(self, split_indexes, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        
        self.dwconv_hw = nn.Conv2d(split_indexes[1], split_indexes[1], square_kernel_size, padding=square_kernel_size//2, groups=split_indexes[1])
        self.dwconv_w = nn.Conv2d(split_indexes[2], split_indexes[2], kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=split_indexes[2])
        self.dwconv_h = nn.Conv2d(split_indexes[3], split_indexes[3], kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=split_indexes[3])
        self.split_indexes = split_indexes
        
    def forward(self, x):
        id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat((id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)    

class InceptionNeXt(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.depthconv = InceptionDWConv2d((in_ch - (in_ch // 8) * 3, in_ch // 8, in_ch // 8, in_ch // 8))
        self.conv1 = nn.Conv2d(in_ch, in_ch * 2, 1)
        self.conv2 = nn.Conv2d(in_ch * 2, in_ch, 1)
        self.act = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.depthconv(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + shortcut
        
class GatedCNNBlock(nn.Module):
    def __init__(self, in_ch, expansion_ratio=2):
        super().__init__()
        self.norm = nn.LayerNorm(in_ch, eps=1e-6)
        hidden = int(expansion_ratio * in_ch)
        self.fc1 = nn.Conv2d(in_ch, hidden * 2, 1)
        self.act = nn.GELU()
        self.conv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.fc2 = nn.Conv2d(hidden, in_ch, 1)

    def forward(self, x):
        shortcut = x
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x1, x2 = self.fc1(x).chunk(2, 1)
        x = self.fc2(self.act(x1) * self.conv(x2))
        return x + shortcut
    
class BasicBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.blocks = nn.Sequential(
            InceptionNeXt(in_ch),
            GatedCNNBlock(in_ch),
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=2, padding=2, groups=out_ch),
        )

    def forward(self, x):
        return self.branch1(x) + self.branch2(x)
    
class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=1, padding=0), 
            nn.PixelShuffle(2),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=5, padding=2, groups=in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=1, padding=0), 
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.branch1(x) + self.branch2(x)
    
class AnalysisTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre1 = Downsample(4, 128)
        self.pre2 = nn.Conv2d(320, 64, kernel_size=3, padding=1)
        self.analysis_transform = nn.Sequential(
            BasicBlock(192),
            Downsample(192, 256),
            BasicBlock(256),
            Downsample(256, 320),
            BasicBlock(320),
        )

    def forward(self, latent, latent2):
        x = torch.cat((self.pre1(latent), self.pre2(latent2)), dim=1)
        x = self.analysis_transform(x)
        return x
    
class SynthesisTransform(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            BasicBlock(320),
            Upsample(320, 320),
            BasicBlock(320),
            Upsample(320, 320),
            BasicBlock(320),
            Upsample(320, 320),
        )

    def forward(self, x):
        x = self.synthesis_transform(x)
        return x
    
class AuxDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(
            BasicBlock(320),
            Upsample(320, 256),
            BasicBlock(256),
            Upsample(256, 192),
            BasicBlock(192),
            Upsample(192, 4),
        )

    def forward(self, x):
        x = self.block(x)
        return x
    
class Adapter(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, (in_ch + out_ch) // 2, 1),
            nn.GELU(),
            nn.Conv2d((in_ch + out_ch) // 2, (in_ch + out_ch) // 2, 5, padding=2, groups=(in_ch + out_ch) // 2),
            nn.GELU(),
            nn.Conv2d((in_ch + out_ch) // 2, out_ch, 1),
        )
        self.branch2 = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.branch1(x) + self.branch2(x)
    
class HyperAnalysis(nn.Module):
    def __init__(self, M=320) -> None:
        super().__init__()
        self.reduction = nn.Sequential(
            nn.Conv2d(M, M // 2, 3, stride=2, padding=1),
            BasicBlock(M // 2),
            nn.Conv2d(M // 2, M // 2, 3, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x
    
class HyperSynthesis(nn.Module):
    def __init__(self, M=320) -> None:
        super().__init__()
        self.increase = nn.Sequential(
            nn.Conv2d(M // 2, M * 2, kernel_size=1, padding=0), 
            nn.PixelShuffle(2),
            BasicBlock(M // 2),
            nn.Conv2d(M // 2, M * 4, kernel_size=1, padding=0), 
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        x = self.increase(x)
        return x
    
class SpatialContext(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.block = nn.Sequential(
            BasicBlock(in_ch),
            BasicBlock(in_ch),
            BasicBlock(in_ch),
            BasicBlock(in_ch),
        )

    def forward(self, x):
        context = self.block(x)
        return context
    
class LRP(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Adapter(in_ch, (in_ch + out_ch) // 2),
            Adapter((in_ch + out_ch) // 2, out_ch),
        )

    def forward(self, x):
        return self.block(x)
    
class LatentCodec(CompressionModel):
    def __init__(self, lambda_rate):
        super().__init__()

        M = 320
        self.g_a = AnalysisTransform()
        self.g_s = SynthesisTransform()
        self.h_a = HyperAnalysis(M=M)
        self.h_s = HyperSynthesis(M=M)
        self.aux = AuxDecoder()

        context_dim = M * 3
        self.adapter_in = nn.ModuleList(Adapter(in_ch=M, out_ch=context_dim) for i in range(4))
        self.g_c = SpatialContext(in_ch=context_dim)
        self.adapter_out = nn.ModuleList(Adapter(in_ch=context_dim, out_ch=M * 2) for i in range(4))
        self.LRP = nn.ModuleList(LRP(in_ch=M * 2, out_ch=M) for i in range(4))

        self.entropy_bottleneck = EntropyBottleneck(M // 2)
        self.gaussian_conditional = GaussianConditional(None)
        self.masks = {}

        self.rate = TargetRateModule(lambda_rate)

    def get_mask_four_parts(self, batch, channel, height, width, device='cuda'):
        curr_mask_str = f"{batch}_{channel}x{width}x{height}"
        if curr_mask_str not in self.masks:
            micro_m0 = torch.tensor(((1., 0), (0, 0)), device=device)
            m0 = micro_m0.repeat((height + 1) // 2, (width + 1) // 2)
            m0 = m0[:height, :width]
            m0 = torch.unsqueeze(m0, 0)
            m0 = torch.unsqueeze(m0, 0)

            micro_m1 = torch.tensor(((0, 1.), (0, 0)), device=device)
            m1 = micro_m1.repeat((height + 1) // 2, (width + 1) // 2)
            m1 = m1[:height, :width]
            m1 = torch.unsqueeze(m1, 0)
            m1 = torch.unsqueeze(m1, 0)

            micro_m2 = torch.tensor(((0, 0), (1., 0)), device=device)
            m2 = micro_m2.repeat((height + 1) // 2, (width + 1) // 2)
            m2 = m2[:height, :width]
            m2 = torch.unsqueeze(m2, 0)
            m2 = torch.unsqueeze(m2, 0)

            micro_m3 = torch.tensor(((0, 0), (0, 1.)), device=device)
            m3 = micro_m3.repeat((height + 1) // 2, (width + 1) // 2)
            m3 = m3[:height, :width]
            m3 = torch.unsqueeze(m3, 0)
            m3 = torch.unsqueeze(m3, 0)

            m = torch.ones((batch, channel // 4, height, width), device=device)
            mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
            mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
            mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
            mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)
            self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]
    
    def sequeeze_with_mask(self, latent, mask):
        latent_group_1, latent_group_2, latent_group_3, latent_group_4 = latent.chunk(4, 1)
        mask_group_1, mask_group_2, mask_group_3, mask_group_4 = mask.chunk(4, 1)
        latent_sequeeze = latent_group_1 * mask_group_1 + latent_group_2 * mask_group_2 + latent_group_3 * mask_group_3 + latent_group_4 * mask_group_4
        return latent_sequeeze
    
    def unsequeeze_with_mask(self, latent_sequeeze, mask):
        mask_group_1, mask_group_2, mask_group_3, mask_group_4 = mask.chunk(4, 1)
        latent = torch.cat((latent_sequeeze * mask_group_1, latent_sequeeze * mask_group_2, latent_sequeeze * mask_group_3, latent_sequeeze * mask_group_4), dim=1)
        return latent
    
    def compress_group_with_mask(self, gaussian_conditional, latent, scales, means, mask, symbols_list, indexes_list):
        latent_squeeze = self.sequeeze_with_mask(latent, mask)
        scales_squeeze = self.sequeeze_with_mask(scales, mask)
        means_squeeze = self.sequeeze_with_mask(means, mask)
        indexes = gaussian_conditional.build_indexes(scales_squeeze)
        latent_squeeze_hat = gaussian_conditional.quantize(latent_squeeze, "symbols", means_squeeze)
        symbols_list.extend(latent_squeeze_hat.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())
        latent_hat = self.unsequeeze_with_mask(latent_squeeze_hat + means_squeeze, mask)
        return latent_hat
    
    def decompress_group_with_mask(self, gaussian_conditional, scales, means, mask, decoder, cdf, cdf_lengths, offsets):
        scales_squeeze = self.sequeeze_with_mask(scales, mask)
        means_squeeze = self.sequeeze_with_mask(means, mask)
        indexes = gaussian_conditional.build_indexes(scales_squeeze)
        latent_squeeze_hat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        latent_squeeze_hat = torch.Tensor(latent_squeeze_hat).reshape(scales_squeeze.shape).to(scales.device)
        latent_hat = self.unsequeeze_with_mask(latent_squeeze_hat + means_squeeze, mask)
        return latent_hat

    def forward(self, latent, latent2, ori_h, ori_w):

        y = self.g_a(latent, latent2)
        z = self.h_a(y)

        _, z_likelihoods = self.entropy_bottleneck(z)
        with torch.no_grad():
            _, quantized_z_likelihoods = self.entropy_bottleneck(z, training=False)

        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset

        B, C, H, W = y.shape
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, device=y.device)

        # Hyper-parameters
        base = self.h_s(z_hat)
        means_0_supp, scales_0_supp = self.adapter_out[0](self.g_c(self.adapter_in[0](base))).chunk(2, 1)
        means_0 = means_0_supp * mask_0
        scales_0 = scales_0_supp * mask_0
        y_0 = y * mask_0
        y_hat_0 = ste_round(y_0 - means_0) + means_0
        lrp = self.LRP[0](torch.cat([y_hat_0, base], dim=1)) * mask_0
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_0 = y_hat_0 + lrp

        base = base * (1 - mask_0) + y_hat_0
        means_1_supp, scales_1_supp = self.adapter_out[1](self.g_c(self.adapter_in[1](base))).chunk(2, 1)
        means_1 = means_1_supp * mask_1
        scales_1 = scales_1_supp * mask_1
        y_1 = y * mask_1
        y_hat_1 = ste_round(y_1 - means_1) + means_1
        lrp = self.LRP[1](torch.cat([y_hat_1, base], dim=1)) * mask_1
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_1 = y_hat_1 + lrp

        base = base * (1 - mask_1) + y_hat_1
        means_2_supp, scales_2_supp = self.adapter_out[2](self.g_c(self.adapter_in[2](base))).chunk(2, 1)
        means_2 = means_2_supp * mask_2
        scales_2 = scales_2_supp * mask_2
        y_2 = y * mask_2
        y_hat_2 = ste_round(y_2 - means_2) + means_2
        lrp = self.LRP[2](torch.cat([y_hat_2, base], dim=1)) * mask_2
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_2 = y_hat_2 + lrp

        base = base * (1 - mask_2) + y_hat_2
        means_3_supp, scales_3_supp = self.adapter_out[3](self.g_c(self.adapter_in[3](base))).chunk(2, 1)
        means_3 = means_3_supp * mask_3
        scales_3 = scales_3_supp * mask_3
        y_3 = y * mask_3
        y_hat_3 = ste_round(y_3 - means_3) + means_3
        lrp = self.LRP[3](torch.cat([y_hat_3, base], dim=1)) * mask_3
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_3 = y_hat_3 + lrp

        scales_all = scales_0 + scales_1 + scales_2 + scales_3
        means_all = means_0 + means_1 + means_2 + means_3

        _, y_likelihoods = self.gaussian_conditional(y, scales_all, means_all)
        with torch.no_grad():
            _, quantized_y_likelihoods = self.gaussian_conditional(y, scales_all, means_all, training=False)

        y_hat = base * (1 - mask_3) + y_hat_3
        x_hat = self.g_s(y_hat)
        res = self.aux(y_hat)

        RateLossOutput = self.rate(
            latent_likelihoods=y_likelihoods,
            quantized_latent_likelihoods=quantized_y_likelihoods,
            hyper_latent_likelihoods=z_likelihoods,
            quantized_hyper_latent_likelihoods=quantized_z_likelihoods,
            ori_h=ori_h,
            ori_w=ori_w,
        )

        return x_hat, RateLossOutput, res

    def compress(self, latent, latent2):

        y = self.g_a(latent, latent2)
        z = self.h_a(y)

        torch.backends.cudnn.deterministic = True
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        B, C, H, W = y.shape
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, device=y.device)

        base = self.h_s(z_hat)
        means_0_supp, scales_0_supp = self.adapter_out[0](self.g_c(self.adapter_in[0](base))).chunk(2, 1)
        y_hat_0 = self.compress_group_with_mask(self.gaussian_conditional, y, scales_0_supp, means_0_supp, mask_0, symbols_list, indexes_list)
        lrp = self.LRP[0](torch.cat([y_hat_0, base], dim=1)) * mask_0
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_0 = y_hat_0 + lrp

        base = base * (1 - mask_0) + y_hat_0
        means_1_supp, scales_1_supp = self.adapter_out[1](self.g_c(self.adapter_in[1](base))).chunk(2, 1)
        y_hat_1 = self.compress_group_with_mask(self.gaussian_conditional, y, scales_1_supp, means_1_supp, mask_1, symbols_list, indexes_list)
        lrp = self.LRP[1](torch.cat([y_hat_1, base], dim=1)) * mask_1
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_1 = y_hat_1 + lrp

        base = base * (1 - mask_1) + y_hat_1
        means_2_supp, scales_2_supp = self.adapter_out[2](self.g_c(self.adapter_in[2](base))).chunk(2, 1)
        y_hat_2 = self.compress_group_with_mask(self.gaussian_conditional, y, scales_2_supp, means_2_supp, mask_2, symbols_list, indexes_list)
        lrp = self.LRP[2](torch.cat([y_hat_2, base], dim=1)) * mask_2
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_2 = y_hat_2 + lrp

        base = base * (1 - mask_2) + y_hat_2
        means_3_supp, scales_3_supp = self.adapter_out[3](self.g_c(self.adapter_in[3](base))).chunk(2, 1)
        _ = self.compress_group_with_mask(self.gaussian_conditional, y, scales_3_supp, means_3_supp, mask_3, symbols_list, indexes_list)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        torch.backends.cudnn.deterministic = False

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, strings, shape):

        torch.backends.cudnn.deterministic = True
        y_strings = strings[0][0]
        z_strings = strings[1]
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        B, C, H, W = z_hat.shape
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C * 2, H * 4, W * 4, device=z_hat.device)

        base = self.h_s(z_hat)
        means_0_supp, scales_0_supp = self.adapter_out[0](self.g_c(self.adapter_in[0](base))).chunk(2, 1)
        y_hat_0 = self.decompress_group_with_mask(self.gaussian_conditional, scales_0_supp, means_0_supp, mask_0, decoder, cdf, cdf_lengths, offsets)
        lrp = self.LRP[0](torch.cat([y_hat_0, base], dim=1)) * mask_0
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_0 = y_hat_0 + lrp

        base = base * (1 - mask_0) + y_hat_0
        means_1_supp, scales_1_supp = self.adapter_out[1](self.g_c(self.adapter_in[1](base))).chunk(2, 1)
        y_hat_1 = self.decompress_group_with_mask(self.gaussian_conditional, scales_1_supp, means_1_supp, mask_1, decoder, cdf, cdf_lengths, offsets)
        lrp = self.LRP[1](torch.cat([y_hat_1, base], dim=1)) * mask_1
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_1 = y_hat_1 + lrp

        base = base * (1 - mask_1) + y_hat_1
        means_2_supp, scales_2_supp = self.adapter_out[2](self.g_c(self.adapter_in[2](base))).chunk(2, 1)
        y_hat_2 = self.decompress_group_with_mask(self.gaussian_conditional, scales_2_supp, means_2_supp, mask_2, decoder, cdf, cdf_lengths, offsets)
        lrp = self.LRP[2](torch.cat([y_hat_2, base], dim=1)) * mask_2
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_2 = y_hat_2 + lrp

        base = base * (1 - mask_2) + y_hat_2
        means_3_supp, scales_3_supp = self.adapter_out[3](self.g_c(self.adapter_in[3](base))).chunk(2, 1)
        y_hat_3 = self.decompress_group_with_mask(self.gaussian_conditional, scales_3_supp, means_3_supp, mask_3, decoder, cdf, cdf_lengths, offsets)
        lrp = self.LRP[3](torch.cat([y_hat_3, base], dim=1)) * mask_3
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_3 = y_hat_3 + lrp

        y_hat = y_hat_0 + y_hat_1 + y_hat_2 + y_hat_3
        torch.backends.cudnn.deterministic = False

        x_hat = self.g_s(y_hat)
        res = self.aux(y_hat)

        return x_hat, res
    
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

class RateLossOutput(NamedTuple):
    rate_loss: Tensor
    quantized_total_bpp: Tensor
    quantized_latent_bpp: Tensor
    quantized_hyper_bpp: Tensor

class TargetRateModule(nn.Module):
    def __init__(self, lambda_rate):
        super().__init__()
        self.lambda_rate = lambda_rate

    def _calc_bits_per_batch(self, likelihoods: Tensor) -> Tensor:
        batch_size = likelihoods.shape[0]
        likelihoods = likelihoods.reshape(batch_size, -1)
        return likelihoods.log().sum(1) / -math.log(2)

    def forward(
        self,
        latent_likelihoods: Tensor,
        quantized_latent_likelihoods: Tensor,
        hyper_latent_likelihoods: Tensor,
        quantized_hyper_latent_likelihoods: Tensor,
        ori_h=512,
        ori_w=512,
    ):
        num_pixels = ori_h * ori_w

        latent_bpp = self._calc_bits_per_batch(latent_likelihoods) / num_pixels
        quantized_latent_bpp = self._calc_bits_per_batch(quantized_latent_likelihoods) / num_pixels
        hyper_bpp = self._calc_bits_per_batch(hyper_latent_likelihoods) / num_pixels
        quantized_hyper_bpp = self._calc_bits_per_batch(quantized_hyper_latent_likelihoods) / num_pixels

        total_bpp = latent_bpp + hyper_bpp
        quantized_total_bpp = quantized_latent_bpp + quantized_hyper_bpp

        return RateLossOutput(
            rate_loss=(self.lambda_rate * total_bpp).mean(),
            quantized_total_bpp=quantized_total_bpp.detach().mean(),
            quantized_latent_bpp=quantized_latent_bpp.detach().mean(),
            quantized_hyper_bpp=quantized_hyper_bpp.detach().mean(),
        )