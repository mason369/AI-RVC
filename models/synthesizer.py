# -*- coding: utf-8 -*-
"""
RVC v2 合成器模型定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class ResidualCouplingBlock(nn.Module):
    """残差耦合块"""

    def __init__(self, channels: int, hidden_channels: int, kernel_size: int,
                 dilation_rate: int, n_layers: int, n_flows: int = 4):
        super().__init__()
        self.flows = nn.ModuleList()

        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels, hidden_channels, kernel_size,
                    dilation_rate, n_layers
                )
            )
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class ResidualCouplingLayer(nn.Module):
    """残差耦合层"""

    def __init__(self, channels: int, hidden_channels: int, kernel_size: int,
                 dilation_rate: int, n_layers: int, mean_only: bool = True):
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers)
        self.post = nn.Conv1d(hidden_channels, self.half_channels, 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, dim=1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        m = stats

        if not reverse:
            x1 = m + x1 * x_mask
            x = torch.cat([x0, x1], dim=1)
            return x, None
        else:
            x1 = (x1 - m) * x_mask
            x = torch.cat([x0, x1], dim=1)
            return x


class Flip(nn.Module):
    """翻转层"""

    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        return x


class WN(nn.Module):
    """WaveNet 风格网络"""

    def __init__(self, hidden_channels: int, kernel_size: int,
                 dilation_rate: int, n_layers: int, gin_channels: int = 0):
        super().__init__()
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()

        if gin_channels > 0:
            self.cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = (kernel_size * dilation - dilation) // 2
            self.in_layers.append(
                nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size,
                          dilation=dilation, padding=padding)
            )
            self.res_skip_layers.append(
                nn.Conv1d(hidden_channels, 2 * hidden_channels, 1)
            )

    def forward(self, x, x_mask, g=None):
        output = torch.zeros_like(x)

        if g is not None and self.gin_channels > 0:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * x.shape[1]
                g_l = g[:, cond_offset:cond_offset + 2 * x.shape[1], :]
                x_in = x_in + g_l

            acts = torch.tanh(x_in[:, :x.shape[1]]) * torch.sigmoid(x_in[:, x.shape[1]:])
            res_skip = self.res_skip_layers[i](acts)

            x = (x + res_skip[:, :x.shape[1]]) * x_mask
            output = output + res_skip[:, x.shape[1]:]

        return output * x_mask


class PosteriorEncoder(nn.Module):
    """后验编码器"""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int,
                 kernel_size: int, dilation_rate: int, n_layers: int,
                 gin_channels: int = 0):
        super().__init__()
        self.out_channels = out_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(
            self._sequence_mask(x_lengths, x.size(2)), 1
        ).to(x.dtype)

        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

    def _sequence_mask(self, length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        return x.unsqueeze(0) < length.unsqueeze(1)


class Generator(nn.Module):
    """HiFi-GAN 生成器"""

    def __init__(self, initial_channel: int, resblock_kernel_sizes: list,
                 resblock_dilation_sizes: list, upsample_rates: list,
                 upsample_initial_channel: int, upsample_kernel_sizes: list,
                 gin_channels: int = 0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, 3)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2 ** i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k, u, (k - u) // 2
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, 3, bias=False)

        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)

        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x


class ResBlock(nn.Module):
    """残差块"""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1,
                      (kernel_size * d - d) // 2, dilation=d)
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1,
                      (kernel_size - 1) // 2)
            for _ in dilation
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class SynthesizerTrnMs768NSFsid(nn.Module):
    """RVC v2 合成器 (768 维 HuBERT + NSF + SID)"""

    def __init__(self, spec_channels: int, segment_size: int,
                 inter_channels: int, hidden_channels: int, filter_channels: int,
                 n_heads: int, n_layers: int, kernel_size: int, p_dropout: float,
                 resblock: str, resblock_kernel_sizes: list,
                 resblock_dilation_sizes: list, upsample_rates: list,
                 upsample_initial_channel: int, upsample_kernel_sizes: list,
                 spk_embed_dim: int, gin_channels: int, sr: int):
        super().__init__()

        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim
        self.sr = sr

        # 编码器
        self.enc_p = PosteriorEncoder(
            768, inter_channels, hidden_channels, 5, 1, 16, gin_channels
        )

        # 解码器/生成器
        self.dec = Generator(
            inter_channels, resblock_kernel_sizes, resblock_dilation_sizes,
            upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
            gin_channels
        )

        # 流
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4
        )

        # 说话人嵌入
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)

    def forward(self, phone, phone_lengths, pitch, nsff0, sid, skip_head=0, return_length=0):
        """前向传播"""
        g = self.emb_g(sid).unsqueeze(-1)

        z, m_p, logs_p, _ = self.enc_p(phone, phone_lengths, g=g)

        z_p = self.flow(z, torch.ones_like(z), g=g)

        # 生成音频
        o = self.dec(z_p, g=g)

        return o

    def infer(self, phone, phone_lengths, pitch, nsff0, sid, rate=1.0):
        """推理"""
        g = self.emb_g(sid).unsqueeze(-1)

        z, m_p, logs_p, x_mask = self.enc_p(phone, phone_lengths, g=g)

        z_p = self.flow(z, x_mask, g=g, reverse=True)

        o = self.dec(z_p * x_mask, g=g)

        return o, x_mask
