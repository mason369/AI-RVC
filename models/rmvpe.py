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

    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU()
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


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
                self._make_encoder_block(
                    in_channels if i == 0 else out_channels * (2 ** (i - 1)),
                    out_channels * (2 ** i),
                    kernel_size,
                    n_blocks,
                    momentum
                )
            )
            self.latent_channels.append(out_channels * (2 ** i))

    def _make_encoder_block(self, in_channels, out_channels, kernel_size,
                            n_blocks, momentum):
        layers = [ConvBlockRes(in_channels, out_channels, momentum)]
        for _ in range(n_blocks - 1):
            layers.append(ConvBlockRes(out_channels, out_channels, momentum))
        layers.append(nn.AvgPool2d(kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bn(x)
        concat_tensors = []
        for layer in self.layers:
            x = layer(x)
            concat_tensors.append(x)
        return x, concat_tensors


class Intermediate(nn.Module):
    """中间层"""

    def __init__(self, in_channels: int, out_channels: int, n_inters: int,
                 n_blocks: int, momentum: float = 0.01):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(ConvBlockRes(in_channels, out_channels, momentum))
        for _ in range(n_inters - 1):
            self.layers.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
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
                self._make_decoder_block(
                    in_ch, out_ch, stride, n_blocks, momentum
                )
            )

    def _make_decoder_block(self, in_channels, out_channels, stride, n_blocks, momentum):
        layers = [nn.ConvTranspose2d(in_channels, out_channels, stride, stride)]
        for _ in range(n_blocks):
            layers.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        return nn.ModuleList(layers)

    def forward(self, x, concat_tensors):
        for i, layer in enumerate(self.layers):
            x = layer[0](x)
            x = torch.cat([x, concat_tensors[-1 - i]], dim=1)
            for block in layer[1:]:
                x = block(x)
        return x


class DeepUnet(nn.Module):
    """Deep U-Net 架构"""

    def __init__(self, kernel_size: int, n_blocks: int, en_de_layers: int = 5,
                 inter_layers: int = 4, in_channels: int = 1, en_out_channels: int = 16):
        super().__init__()

        self.encoder = Encoder(
            in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels
        )
        self.intermediate = Intermediate(
            en_out_channels * (2 ** (en_de_layers - 1)),
            en_out_channels * (2 ** (en_de_layers - 1)),
            inter_layers, n_blocks
        )
        self.decoder = Decoder(
            en_out_channels * (2 ** (en_de_layers - 1)),
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

    def forward(self, x):
        x = self.unet(x)
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(2)
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
        mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
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
        spec = torch.abs(spec)

        # Mel 变换
        mel = torch.matmul(self.mel_basis, spec)
        mel = torch.log(torch.clamp(mel, min=1e-5))

        return mel


class RMVPE:
    """RMVPE F0 提取器封装类"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device

        # 加载模型
        self.model = E2E(n_blocks=4, n_gru=2, kernel_size=2)
        ckpt = torch.load(model_path, map_location="cpu")
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

        # 提取 Mel 频谱
        mel = self.mel_extractor(audio)
        mel = mel.unsqueeze(1)

        # 模型推理
        hidden = self.model(mel)
        hidden = hidden.squeeze(0).cpu().numpy()

        # 解码 F0
        f0 = self._decode(hidden, thred)

        return f0

    def _decode(self, hidden: np.ndarray, thred: float) -> np.ndarray:
        """解码隐藏状态为 F0"""
        # 找到最大值位置
        center = np.argmax(hidden, axis=1)
        hidden = np.pad(hidden, ((0, 0), (4, 4)), mode="constant")

        # 加权平均
        f0 = []
        for i, c in enumerate(center):
            c += 4
            weights = hidden[i, c - 4:c + 5]
            if weights.sum() < thred:
                f0.append(0)
            else:
                cents = self.cents_mapping[c - 4:c + 5]
                f0.append(np.sum(weights * cents) / weights.sum())

        f0 = np.array(f0)
        f0 = 10 * (2 ** (f0 / 1200))
        f0[f0 < 10] = 0

        return f0
