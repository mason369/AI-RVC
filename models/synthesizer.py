# -*- coding: utf-8 -*-
"""
RVC v2 合成器模型定义
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class LayerNorm(nn.Module):
    """Layer normalization for channels-first tensors"""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        # x: [B, C, T]
        x = x.transpose(1, -1)  # [B, T, C]
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)  # [B, C, T]


class MultiHeadAttention(nn.Module):
    """Multi-head attention module"""

    def __init__(self, channels: int, out_channels: int, n_heads: int,
                 p_dropout: float = 0.0, window_size: Optional[int] = None,
                 heads_share: bool = True, block_length: Optional[int] = None,
                 proximal_bias: bool = False, proximal_init: bool = False):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels ** -0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev
            )

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        # query, key, value: [B, C, T]
        b, d, t_s = key.size()
        t_t = query.size(2)

        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))

        if self.window_size is not None:
            assert t_s == t_t, "Relative attention only for self-attention"
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(query / math.sqrt(self.k_channels), key_relative_embeddings)
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local

        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias only for self-attention"
            scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                assert t_s == t_t, "Block length only for self-attention"
                block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
                scores = scores.masked_fill(block_mask == 0, -1e4)

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)

        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)

        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        max_relative_position = 2 * self.window_size + 1
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                (0, 0, pad_length, pad_length, 0, 0)
            )
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[:, slice_start_position:slice_end_position]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, (0, 1, 0, 0, 0, 0, 0, 0))
        x_flat = x.view(batch, heads, length * 2 * length)
        x_flat = F.pad(x_flat, (0, length - 1, 0, 0, 0, 0))
        x_final = x_flat.view(batch, heads, length + 1, 2 * length - 1)[:, :, :length, length - 1:]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, (0, length - 1, 0, 0, 0, 0, 0, 0))
        x_flat = x.view(batch, heads, length ** 2 + length * (length - 1))
        x_flat = F.pad(x_flat, (length, 0, 0, 0, 0, 0))
        x_final = x_flat.view(batch, heads, length, 2 * length)[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
    """Feed-forward network with optional causal convolution"""

    def __init__(self, in_channels: int, out_channels: int, filter_channels: int,
                 kernel_size: int, p_dropout: float = 0.0, activation: str = None,
                 causal: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x))
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self.padding(x))
        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        return F.pad(x, (pad_l, pad_r, 0, 0, 0, 0))

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        return F.pad(x, (pad_l, pad_r, 0, 0, 0, 0))


class Encoder(nn.Module):
    """Transformer encoder with multi-head attention"""

    def __init__(self, hidden_channels: int, filter_channels: int, n_heads: int,
                 n_layers: int, kernel_size: int = 1, p_dropout: float = 0.0,
                 window_size: int = 10):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for _ in range(n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels, hidden_channels, n_heads,
                    p_dropout=p_dropout, window_size=window_size
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(hidden_channels, hidden_channels, filter_channels,
                    kernel_size, p_dropout=p_dropout)
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class TextEncoder(nn.Module):
    """Text encoder for RVC - encodes phone and pitch embeddings"""

    def __init__(self, out_channels: int, hidden_channels: int, filter_channels: int,
                 n_heads: int, n_layers: int, kernel_size: int, p_dropout: float,
                 f0: bool = True):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.f0 = f0

        # Phone embedding: Linear projection from 768-dim HuBERT features
        self.emb_phone = nn.Linear(768, hidden_channels)

        # Pitch embedding (only if f0 is enabled)
        if f0:
            self.emb_pitch = nn.Embedding(256, hidden_channels)

        # Transformer encoder
        self.encoder = Encoder(
            hidden_channels, filter_channels, n_heads, n_layers,
            kernel_size, p_dropout
        )

        # Output projection to mean and log-variance
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, phone, pitch, lengths):
        """
        Args:
            phone: [B, 768, T] phone features from HuBERT (channels first)
            pitch: [B, T] pitch indices (0-255)
            lengths: [B] sequence lengths

        Returns:
            m: [B, out_channels, T] mean
            logs: [B, out_channels, T] log-variance
            x_mask: [B, 1, T] mask
        """
        import logging
        log = logging.getLogger(__name__)

        log.debug(f"[TextEncoder] 输入 phone: shape={phone.shape}")
        log.debug(f"[TextEncoder] 输入 pitch: shape={pitch.shape}, max={pitch.max().item()}, min={pitch.min().item()}")
        log.debug(f"[TextEncoder] 输入 lengths: {lengths}")

        # Transpose phone from [B, C, T] to [B, T, C] for linear layer
        phone = phone.transpose(1, 2)  # [B, T, 768]
        log.debug(f"[TextEncoder] 转置后 phone: shape={phone.shape}")

        # Create mask
        x_mask = torch.unsqueeze(
            self._sequence_mask(lengths, phone.size(1)), 1
        ).to(phone.dtype)
        log.debug(f"[TextEncoder] x_mask: shape={x_mask.shape}, sum={x_mask.sum().item()}")

        # Phone embedding
        x = self.emb_phone(phone)  # [B, T, hidden_channels]
        log.debug(f"[TextEncoder] emb_phone 输出: shape={x.shape}, max={x.abs().max().item():.4f}, mean={x.abs().mean().item():.4f}")

        # Add pitch embedding if enabled
        if self.f0 and pitch is not None:
            # Clamp pitch to valid range
            pitch_clamped = torch.clamp(pitch, 0, 255)
            pitch_emb = self.emb_pitch(pitch_clamped)
            log.debug(f"[TextEncoder] emb_pitch 输出: shape={pitch_emb.shape}, max={pitch_emb.abs().max().item():.4f}")
            x = x + pitch_emb

        # Transpose for conv layers: [B, hidden_channels, T]
        x = x.transpose(1, 2)
        log.debug(f"[TextEncoder] 转置后 x: shape={x.shape}")

        # Apply mask
        x = x * x_mask

        # Transformer encoder
        x = self.encoder(x, x_mask)
        log.debug(f"[TextEncoder] Transformer 输出: shape={x.shape}, max={x.abs().max().item():.4f}, mean={x.abs().mean().item():.4f}")

        # Project to mean and log-variance
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        log.debug(f"[TextEncoder] 最终输出 m: shape={m.shape}, max={m.abs().max().item():.4f}")
        log.debug(f"[TextEncoder] 最终输出 logs: shape={logs.shape}, max={logs.max().item():.4f}, min={logs.min().item():.4f}")

        return m, logs, x_mask

    def _sequence_mask(self, length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        return x.unsqueeze(0) < length.unsqueeze(1)


class ResidualCouplingBlock(nn.Module):
    """残差耦合块"""

    def __init__(self, channels: int, hidden_channels: int, kernel_size: int,
                 dilation_rate: int, n_layers: int, n_flows: int = 4,
                 gin_channels: int = 0):
        super().__init__()
        self.flows = nn.ModuleList()

        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels, hidden_channels, kernel_size,
                    dilation_rate, n_layers, gin_channels=gin_channels
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
                 dilation_rate: int, n_layers: int, mean_only: bool = True,
                 gin_channels: int = 0):
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels)
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
    """WaveNet 风格网络 (带权重归一化)"""

    def __init__(self, hidden_channels: int, kernel_size: int,
                 dilation_rate: int, n_layers: int, gin_channels: int = 0,
                 p_dropout: float = 0):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        self.gin_channels = gin_channels

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels > 0:
            self.cond_layer = nn.utils.weight_norm(
                nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            )

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = (kernel_size * dilation - dilation) // 2
            self.in_layers.append(
                nn.utils.weight_norm(
                    nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size,
                              dilation=dilation, padding=padding)
                )
            )
            # 前 n-1 层输出 2 * hidden_channels，最后一层输出 hidden_channels
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels
            self.res_skip_layers.append(
                nn.utils.weight_norm(
                    nn.Conv1d(hidden_channels, res_skip_channels, 1)
                )
            )

    def forward(self, x, x_mask, g=None):
        output = torch.zeros_like(x)

        if g is not None and self.gin_channels > 0:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
                x_in = x_in + g_l

            acts = torch.tanh(x_in[:, :self.hidden_channels]) * torch.sigmoid(x_in[:, self.hidden_channels:])
            acts = self.drop(acts)
            res_skip = self.res_skip_layers[i](acts)

            if i < self.n_layers - 1:
                # 前 n-1 层：residual + skip
                x = (x + res_skip[:, :self.hidden_channels]) * x_mask
                output = output + res_skip[:, self.hidden_channels:]
            else:
                # 最后一层：只有 residual，加到 output
                x = (x + res_skip) * x_mask
                output = output + res_skip

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
    """NSF-HiFi-GAN 生成器 (带权重归一化)"""

    def __init__(self, initial_channel: int, resblock_kernel_sizes: list,
                 resblock_dilation_sizes: list, upsample_rates: list,
                 upsample_initial_channel: int, upsample_kernel_sizes: list,
                 gin_channels: int = 0, sr: int = 40000, is_half: bool = False):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.sr = sr
        self.is_half = is_half

        # 计算上采样因子
        self.upp = int(np.prod(upsample_rates))

        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, 3)

        # NSF 源模块
        self.m_source = SourceModuleHnNSF(sample_rate=sr, harmonic_num=0)

        # 噪声卷积层
        self.noise_convs = nn.ModuleList()

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                nn.utils.weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        c_cur,
                        k, u, (k - u) // 2
                    )
                )
            )
            # 噪声卷积
            if i + 1 < len(upsample_rates):
                stride_f0 = int(np.prod(upsample_rates[i + 1:]))
                self.noise_convs.append(
                    nn.Conv1d(1, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=stride_f0 // 2)
                )
            else:
                self.noise_convs.append(nn.Conv1d(1, c_cur, kernel_size=1))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, 3, bias=False)

        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, f0, g=None):
        import logging
        log = logging.getLogger(__name__)

        log.debug(f"[Generator] 输入 x: shape={x.shape}, max={x.abs().max().item():.4f}, mean={x.abs().mean().item():.4f}")
        log.debug(f"[Generator] 输入 f0: shape={f0.shape}, max={f0.max().item():.1f}, min={f0.min().item():.1f}")
        if g is not None:
            log.debug(f"[Generator] 输入 g: shape={g.shape}, max={g.abs().max().item():.4f}")

        # 生成 NSF 激励信号
        har_source, _, _ = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)  # [B, 1, T*upp]
        log.debug(f"[Generator] NSF har_source: shape={har_source.shape}, max={har_source.abs().max().item():.4f}")

        x = self.conv_pre(x)
        log.debug(f"[Generator] conv_pre 输出: shape={x.shape}, max={x.abs().max().item():.4f}")

        if g is not None:
            x = x + self.cond(g)
            log.debug(f"[Generator] 加入条件后: max={x.abs().max().item():.4f}")

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)

            # 融合噪声
            x_source = self.noise_convs[i](har_source)
            x = x + x_source

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
            log.debug(f"[Generator] 上采样层 {i}: shape={x.shape}, max={x.abs().max().item():.4f}")

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        log.debug(f"[Generator] conv_post 输出: shape={x.shape}, max={x.abs().max().item():.4f}")
        x = torch.tanh(x)
        log.debug(f"[Generator] tanh 输出: shape={x.shape}, max={x.abs().max().item():.4f}")

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            nn.utils.remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class ResBlock(nn.Module):
    """残差块 (带权重归一化)"""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.utils.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, 1,
                          (kernel_size * d - d) // 2, dilation=d)
            )
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            nn.utils.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, 1,
                          (kernel_size - 1) // 2)
            )
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

    def remove_weight_norm(self):
        for l in self.convs1:
            nn.utils.remove_weight_norm(l)
        for l in self.convs2:
            nn.utils.remove_weight_norm(l)


class SineGenerator(nn.Module):
    """正弦波生成器 - NSF 的核心组件"""

    def __init__(self, sample_rate: int, harmonic_num: int = 0,
                 sine_amp: float = 0.1, noise_std: float = 0.003,
                 voiced_threshold: float = 10):
        super().__init__()
        self.sample_rate = sample_rate
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold
        self.dim = harmonic_num + 1

    def forward(self, f0: torch.Tensor, upp: int):
        """
        生成正弦波激励信号

        Args:
            f0: 基频张量 [B, T]
            upp: 上采样因子

        Returns:
            正弦波信号 [B, T*upp, 1]
        """
        with torch.no_grad():
            # 上采样 F0
            f0 = f0.unsqueeze(1)  # [B, 1, T]
            f0_up = F.interpolate(f0, scale_factor=upp, mode='nearest')
            f0_up = f0_up.transpose(1, 2)  # [B, T*upp, 1]

            # 生成正弦波
            rad = f0_up / self.sample_rate  # 归一化频率
            rad_acc = torch.cumsum(rad, dim=1) % 1  # 累积相位
            sine_wave = torch.sin(2 * np.pi * rad_acc) * self.sine_amp

            # 静音区域（F0=0）使用噪声
            voiced_mask = (f0_up > self.voiced_threshold).float()
            noise = torch.randn_like(sine_wave) * self.noise_std
            sine_wave = sine_wave * voiced_mask + noise * (1 - voiced_mask)

            return sine_wave


class SourceModuleHnNSF(nn.Module):
    """谐波加噪声源模块"""

    def __init__(self, sample_rate: int, harmonic_num: int = 0,
                 sine_amp: float = 0.1, noise_std: float = 0.003,
                 add_noise_std: float = 0.003):
        super().__init__()
        self.sine_generator = SineGenerator(
            sample_rate, harmonic_num, sine_amp, noise_std
        )
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, f0: torch.Tensor, upp: int):
        sine = self.sine_generator(f0, upp)  # [B, T*upp, 1]
        sine = self.l_tanh(self.l_linear(sine))
        noise = torch.randn_like(sine) * 0.003
        return sine, noise, None  # 返回 3 个值以匹配接口


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

        # 文本编码器 (使用 TextEncoder 替代 PosteriorEncoder)
        self.enc_p = TextEncoder(
            inter_channels, hidden_channels, filter_channels,
            n_heads, n_layers, kernel_size, p_dropout, f0=True
        )

        # 解码器/生成器 (NSF-HiFiGAN，内部包含 m_source)
        self.dec = Generator(
            inter_channels, resblock_kernel_sizes, resblock_dilation_sizes,
            upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
            gin_channels, sr=sr
        )

        # 流
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
        )

        # 说话人嵌入
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)

    def forward(self, phone, phone_lengths, pitch, nsff0, sid, skip_head=0, return_length=0):
        """前向传播"""
        g = self.emb_g(sid).unsqueeze(-1)

        # TextEncoder 返回 mean 和 log-variance
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)

        # 在编码器外部采样
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask

        # 正向 flow
        z = self.flow(z_p, x_mask, g=g)

        # 生成音频 (传入 f0)
        o = self.dec(z, nsff0, g=g)

        return o

    def infer(self, phone, phone_lengths, pitch, nsff0, sid, rate=1.0):
        """推理"""
        import logging
        log = logging.getLogger(__name__)

        log.debug(f"[infer] 输入 phone: shape={phone.shape}, dtype={phone.dtype}")
        log.debug(f"[infer] 输入 phone 统计: max={phone.abs().max().item():.4f}, mean={phone.abs().mean().item():.4f}")
        log.debug(f"[infer] 输入 phone_lengths: {phone_lengths}")
        log.debug(f"[infer] 输入 pitch: shape={pitch.shape}, max={pitch.max().item()}, min={pitch.min().item()}")
        log.debug(f"[infer] 输入 nsff0: shape={nsff0.shape}, max={nsff0.max().item():.1f}, min={nsff0.min().item():.1f}")
        log.debug(f"[infer] 输入 sid: {sid}")

        g = self.emb_g(sid).unsqueeze(-1)
        log.debug(f"[infer] 说话人嵌入 g: shape={g.shape}, max={g.abs().max().item():.4f}")

        # TextEncoder 返回 mean 和 log-variance
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        log.debug(f"[infer] TextEncoder 输出:")
        log.debug(f"[infer]   m_p: shape={m_p.shape}, max={m_p.abs().max().item():.4f}, mean={m_p.abs().mean().item():.4f}")
        log.debug(f"[infer]   logs_p: shape={logs_p.shape}, max={logs_p.max().item():.4f}, min={logs_p.min().item():.4f}")
        log.debug(f"[infer]   x_mask: shape={x_mask.shape}, sum={x_mask.sum().item()}")

        # 在编码器外部采样 (使用较小的噪声系数以获得更稳定的输出)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        log.debug(f"[infer] 采样后 z_p: shape={z_p.shape}, max={z_p.abs().max().item():.4f}, mean={z_p.abs().mean().item():.4f}")

        # 反向 flow
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        log.debug(f"[infer] Flow 输出 z: shape={z.shape}, max={z.abs().max().item():.4f}, mean={z.abs().mean().item():.4f}")

        # 生成音频 (传入 f0，Generator 内部会生成 NSF 激励信号)
        o = self.dec(z * x_mask, nsff0, g=g)
        log.debug(f"[infer] Generator 输出 o: shape={o.shape}, max={o.abs().max().item():.4f}, mean={o.abs().mean().item():.4f}")

        return o, x_mask
