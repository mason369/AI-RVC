# -*- coding: utf-8 -*-
"""
RMVPE 模型 - 用于高质量 F0 提取
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class BiGRU(nn.Module):
    """双向 GRU 层"""

    def __init__(self, input_features: int, hidden_features: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    """残差卷积块"""

    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.01,
                 force_shortcut: bool = False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU()
        )

        # 当通道数不同或强制使用时才创建 shortcut
        if in_channels != out_channels or force_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
            self.has_shortcut = True
        else:
            self.has_shortcut = False

    def forward(self, x):
        if self.has_shortcut:
            return self.conv(x) + self.shortcut(x)
        else:
            return self.conv(x) + x


class EncoderBlock(nn.Module):
    """编码器块 - 包含多个 ConvBlockRes 和一个池化层"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 n_blocks: int, momentum: float = 0.01):
        super().__init__()
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.pool = nn.AvgPool2d(kernel_size)

    def forward(self, x):
        for block in self.conv:
            x = block(x)
        # 返回池化前的张量用于 skip connection
        return self.pool(x), x


class Encoder(nn.Module):
    """RMVPE 编码器"""

    def __init__(self, in_channels: int, in_size: int, n_encoders: int,
                 kernel_size: int, n_blocks: int, out_channels: int = 16,
                 momentum: float = 0.01):
        super().__init__()

        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []

        for i in range(n_encoders):
            self.layers.append(
                EncoderBlock(
                    in_channels if i == 0 else out_channels * (2 ** (i - 1)),
                    out_channels * (2 ** i),
                    kernel_size,
                    n_blocks,
                    momentum
                )
            )
            self.latent_channels.append(out_channels * (2 ** i))

    def forward(self, x):
        x = self.bn(x)
        concat_tensors = []
        for layer in self.layers:
            x, skip = layer(x)
            concat_tensors.append(skip)
        return x, concat_tensors


class Intermediate(nn.Module):
    """中间层"""

    def __init__(self, in_channels: int, out_channels: int, n_inters: int,
                 n_blocks: int, momentum: float = 0.01):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(n_inters):
            if i == 0:
                # 第一层: in_channels -> out_channels (256 -> 512)
                self.layers.append(
                    IntermediateBlock(in_channels, out_channels, n_blocks, momentum, first_block_shortcut=True)
                )
            else:
                # 后续层: out_channels -> out_channels (512 -> 512)
                self.layers.append(
                    IntermediateBlock(out_channels, out_channels, n_blocks, momentum, first_block_shortcut=False)
                )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class IntermediateBlock(nn.Module):
    """中间层块"""

    def __init__(self, in_channels: int, out_channels: int, n_blocks: int,
                 momentum: float = 0.01, first_block_shortcut: bool = False):
        super().__init__()
        self.conv = nn.ModuleList()
        # 第一个块可能需要强制使用 shortcut
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum, force_shortcut=first_block_shortcut))
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x):
        for block in self.conv:
            x = block(x)
        return x


class DecoderBlock(nn.Module):
    """解码器块"""

    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 n_blocks: int, momentum: float = 0.01):
        super().__init__()
        # conv1: 转置卷积 + BatchNorm (kernel_size=3, stride=stride, padding=1, output_padding=1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum)
        )
        # conv2: ConvBlockRes 列表
        # 第一个块: in_channels = out_channels * 2 (concat 后), out_channels = out_channels
        # 后续块: in_channels = out_channels, out_channels = out_channels
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        # 处理尺寸不匹配：填充较小的张量使其匹配较大的
        diff_h = concat_tensor.size(2) - x.size(2)
        diff_w = concat_tensor.size(3) - x.size(3)
        if diff_h != 0 or diff_w != 0:
            # 填充 x 使其与 concat_tensor 尺寸匹配
            x = F.pad(x, [0, diff_w, 0, diff_h])
        x = torch.cat([x, concat_tensor], dim=1)
        for block in self.conv2:
            x = block(x)
        return x


class Decoder(nn.Module):
    """RMVPE 解码器"""

    def __init__(self, in_channels: int, n_decoders: int, stride: int,
                 n_blocks: int, out_channels: int = 16, momentum: float = 0.01):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(n_decoders):
            out_ch = out_channels * (2 ** (n_decoders - 1 - i))
            in_ch = in_channels if i == 0 else out_channels * (2 ** (n_decoders - i))
            self.layers.append(
                DecoderBlock(in_ch, out_ch, stride, n_blocks, momentum)
            )

    def forward(self, x, concat_tensors):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    """Deep U-Net 架构"""

    def __init__(self, kernel_size: int, n_blocks: int, en_de_layers: int = 5,
                 inter_layers: int = 4, in_channels: int = 1, en_out_channels: int = 16):
        super().__init__()

        # Encoder 输出通道: en_out_channels * 2^(en_de_layers-1) = 16 * 16 = 256
        encoder_out_channels = en_out_channels * (2 ** (en_de_layers - 1))
        # Intermediate 输出通道: encoder_out_channels * 2 = 512
        intermediate_out_channels = encoder_out_channels * 2

        self.encoder = Encoder(
            in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels
        )
        self.intermediate = Intermediate(
            encoder_out_channels,
            intermediate_out_channels,
            inter_layers, n_blocks
        )
        self.decoder = Decoder(
            intermediate_out_channels,
            en_de_layers, kernel_size, n_blocks, en_out_channels
        )

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class E2E(nn.Module):
    """端到端 RMVPE 模型"""

    def __init__(self, n_blocks: int, n_gru: int, kernel_size: int,
                 en_de_layers: int = 5, inter_layers: int = 4,
                 in_channels: int = 1, en_out_channels: int = 16):
        super().__init__()

        self.unet = DeepUnet(
            kernel_size, n_blocks, en_de_layers, inter_layers,
            in_channels, en_out_channels
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, 3, 1, 1)

        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * 128, 256, n_gru),
                nn.Linear(512, 360),
                nn.Dropout(0.25),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * 128, 360),
                nn.Dropout(0.25),
                nn.Sigmoid()
            )

    def forward(self, mel):
        # 输入 mel: [B, 128, T] 或 [B, 1, 128, T]
        # 官方实现期望 [B, 1, T, 128]，即 time 在 height，mel bins 在 width
        if mel.dim() == 3:
            # [B, 128, T] -> [B, T, 128] -> [B, 1, T, 128]
            mel = mel.transpose(-1, -2).unsqueeze(1)
        elif mel.dim() == 4 and mel.shape[1] == 1:
            # [B, 1, 128, T] -> [B, 1, T, 128]
            mel = mel.transpose(-1, -2)

        x = self.unet(mel)
        x = self.cnn(x)
        # x shape: (batch, 3, T, 128)
        # 转换为 (batch, T, 384) 其中 384 = 3 * 128
        x = x.transpose(1, 2).flatten(-2)  # (batch, T, 384)
        x = self.fc(x)
        return x


class MelSpectrogram(nn.Module):
    """Mel 频谱提取"""

    def __init__(self, n_mel: int = 128, n_fft: int = 1024, win_size: int = 1024,
                 hop_length: int = 160, sample_rate: int = 16000,
                 fmin: int = 30, fmax: int = 8000):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_size = win_size
        self.sample_rate = sample_rate
        self.n_mel = n_mel

        # 创建 Mel 滤波器组
        mel_basis = self._mel_filterbank(sample_rate, n_fft, n_mel, fmin, fmax)
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", torch.hann_window(win_size))

    def _mel_filterbank(self, sr, n_fft, n_mels, fmin, fmax):
        """创建 Mel 滤波器组"""
        import librosa
        # 必须使用 htk=True，与官方 RVC RMVPE 保持一致
        mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=True)
        return torch.from_numpy(mel).float()

    def forward(self, audio):
        # STFT
        spec = torch.stft(
            audio,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_size,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True
        )
        # 使用功率谱（幅度的平方），与官方 RMVPE 一致
        spec = torch.abs(spec) ** 2

        # Mel 变换
        mel = torch.matmul(self.mel_basis, spec)
        mel = torch.log(torch.clamp(mel, min=1e-5))

        return mel


class RMVPE:
    """RMVPE F0 提取器封装类"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device

        # 加载模型
        self.model = E2E(n_blocks=4, n_gru=1, kernel_size=2)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt)
        self.model = self.model.to(device).eval()

        # Mel 频谱提取器
        self.mel_extractor = MelSpectrogram().to(device)

        # 频率映射
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))

    @torch.no_grad()
    def infer_from_audio(self, audio: np.ndarray, thred: float = 0.03) -> np.ndarray:
        """
        从音频提取 F0

        Args:
            audio: 16kHz 音频数据
            thred: 置信度阈值

        Returns:
            np.ndarray: F0 序列
        """
        # 转换为张量
        audio = torch.from_numpy(audio).float().to(self.device)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # 提取 Mel 频谱: [B, 128, T]
        mel = self.mel_extractor(audio)

        # 记录原始帧数
        n_frames = mel.shape[-1]

        # 填充时间维度使其可被 32 整除（5 层池化，每层 /2）
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad > 0:
            mel = F.pad(mel, (0, n_pad), mode='constant', value=0)

        # 模型推理 - E2E.forward 会处理 transpose
        hidden = self.model(mel)

        # 移除填充部分，只保留原始帧数
        hidden = hidden[:, :n_frames, :]
        hidden = hidden.squeeze(0).cpu().numpy()

        # 解码 F0
        f0 = self._decode(hidden, thred)

        return f0

    def _decode(self, hidden: np.ndarray, thred: float) -> np.ndarray:
        """解码隐藏状态为 F0 - 使用官方 RVC 算法"""
        # 使用官方的 to_local_average_cents 算法
        cents = self._to_local_average_cents(hidden, thred)

        # 转换 cents 到 Hz
        f0 = 10 * (2 ** (cents / 1200))
        f0[f0 == 10] = 0  # cents=0 时 f0=10，需要置零

        return f0

    def _to_local_average_cents(self, salience: np.ndarray, thred: float) -> np.ndarray:
        """官方 RVC 的 to_local_average_cents 算法"""
        # Step 1: 找到每帧的峰值 bin
        center = np.argmax(salience, axis=1)  # [T]

        # Step 2: 对 salience 进行 padding
        salience = np.pad(salience, ((0, 0), (4, 4)))  # [T, 368]
        center += 4  # 调整 center 索引

        # Step 3: 提取峰值附近 9 个 bin 的窗口并计算加权平均
        todo_salience = []
        todo_cents_mapping = []
        starts = center - 4
        ends = center + 5

        for idx in range(salience.shape[0]):
            todo_salience.append(salience[idx, starts[idx]:ends[idx]])
            todo_cents_mapping.append(self.cents_mapping[starts[idx]:ends[idx]])

        todo_salience = np.array(todo_salience)  # [T, 9]
        todo_cents_mapping = np.array(todo_cents_mapping)  # [T, 9]

        # Step 4: 加权平均
        product_sum = np.sum(todo_salience * todo_cents_mapping, axis=1)
        weight_sum = np.sum(todo_salience, axis=1) + 1e-9
        cents = product_sum / weight_sum

        # Step 5: 阈值过滤 - 使用原始 salience 的最大值
        maxx = np.max(salience, axis=1)
        cents[maxx <= thred] = 0

        return cents
