"""The Vocos decoder of indic-speak: SNAC latents [B, 768, L] -> audio [B, 1, 512 L] at 24 kHz.

Vocos (Siuzdak 2023, arXiv:2306.00814) makes audio from ConvNeXt blocks and an inverse STFT. indic-speak's file
(vocos/best.pt, pinned in models.py) has its layout in `config.model`: ConvNeXt blocks at the latent rate, a nearest
upsample, more blocks, a layer norm and an iSTFT head. This module builds that layout. The parameter names match
the file, so a strict load checks every name and shape. No code from the download runs.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

LATENT_DIM = 768        # the SNAC 24 kHz latent
SAMPLES_PER_STEP = 512  # SNAC samples per latent step
CONFIG_KEYS = ("dim", "intermediate_dim", "num_blocks_pre", "num_blocks_post", "upsample", "n_fft", "hop")


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, scale: float):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden, dim)
        self.gamma = nn.Parameter(torch.full((dim,), scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, T]
        y = self.norm(self.dwconv(x).transpose(1, 2))
        y = self.gamma * self.pwconv2(self.act(self.pwconv1(y)))
        return x + y.transpose(1, 2)


class ISTFTHead(nn.Module):
    """Magnitude (exp, at most e^6) and phase per bin, then a centered inverse STFT."""

    def __init__(self, dim: int, n_fft: int, hop: int):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.out = nn.Linear(dim, n_fft + 2)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, T] -> [B, 1, hop T]
        x = F.pad(x, (0, 1), mode="replicate")  # one more frame: the centered iSTFT then gives exactly hop * T samples
        mag, phase = self.out(x.transpose(1, 2)).transpose(1, 2).float().chunk(2, dim=1)
        spec = torch.polar(torch.exp(mag.clamp(max=6.0)), phase)
        audio = torch.istft(spec, self.n_fft, hop_length=self.hop, win_length=self.n_fft, window=self.window.float(),
                            center=True)
        return audio.unsqueeze(1)


class VocosDecoder(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, num_blocks_pre: int, num_blocks_post: int, upsample: int,
                 n_fft: int, hop: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        if upsample * hop != SAMPLES_PER_STEP:
            raise ValueError(f"upsample x hop must be {SAMPLES_PER_STEP}, not {upsample * hop}.")
        scale = 1.0 / (num_blocks_pre + num_blocks_post)
        self.upsample = upsample
        self.stem = nn.Conv1d(latent_dim, dim, kernel_size=7, padding=3)
        self.pre = nn.ModuleList(ConvNeXtBlock(dim, intermediate_dim, scale) for _ in range(num_blocks_pre))
        self.up_conv = nn.Conv1d(dim, dim, kernel_size=7, padding=3)
        self.post = nn.ModuleList(ConvNeXtBlock(dim, intermediate_dim, scale) for _ in range(num_blocks_post))
        self.final_norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = ISTFTHead(dim, n_fft, hop)

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # [B, 768, L] -> [B, 1, 512 L]
        x = self.stem(z)
        for block in self.pre:
            x = block(x)
        x = self.up_conv(x.repeat_interleave(self.upsample, dim=-1))
        for block in self.post:
            x = block(x)
        x = self.final_norm(x.transpose(1, 2)).transpose(1, 2)
        return self.head(x)


def load(path: Path, device: str = "cpu") -> VocosDecoder:
    """The decoder from indic-speak's checkpoint: its own config, its EMA weights (the inference weights).
    The file is a pickle, so it is loaded with weights_only=True: only tensors and plain data can come out of it."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ck["config"]["model"]
    if cfg.get("head_type", "istft") != "istft" or cfg.get("noise_inject"):
        raise ValueError(f"{path}: this Vocos layout ({cfg}) is not the one this module builds.")
    dec = VocosDecoder(**{k: cfg[k] for k in CONFIG_KEYS})
    dec.load_state_dict(ck.get("ema") or ck["vocos"])
    return dec.eval().to(device)
