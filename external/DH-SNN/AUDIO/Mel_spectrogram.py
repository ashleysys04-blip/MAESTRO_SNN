import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torchaudio


# =========================================================
# Config
# =========================================================
@dataclass
class AudioSpikeConfig:
    # audio frontend
    sr: int = 16000
    n_mels: int = 64
    hop_length: int = 160      # 10 ms @ 16 kHz
    n_fft: int = 1024

    # sample length (FULL SEQUENCE)
    clip_seconds: float = 10.0   # ← 핵심: 10초 전체 시퀀스

    # style change location (seconds)
    # None → random 30~70%
    change_seconds: Optional[float] = None

    # spike encoding thresholds
    theta_on: float = 0.15
    theta_off: float = -0.15

    # dataset paths
    baroque_dir: str = None
    romantic_dir: str = None

    # batch size
    batch_size: int = 64


# =========================================================
# Utils
# =========================================================
def _default_data_dirs() -> Tuple[str, str]:
    here = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(here, "data")
    return (
        os.path.join(data_root, "baroque"),
        os.path.join(data_root, "romantic"),
    )


def _list_audio_files(folder: str) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    files = []
    for fn in os.listdir(folder):
        if fn.lower().endswith((".wav", ".mp3")):
            files.append(os.path.join(folder, fn))
    files.sort()
    return files


# =========================================================
# Audio → log-mel
# =========================================================
class AudioToLogMel:
    def __init__(self, cfg: AudioSpikeConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            power=2.0,
        ).to(device)

    def load_mono_resample(self, path: str) -> torch.Tensor:
        wav, orig_sr = torchaudio.load(path)  # (C, N)
        wav = wav.to(self.device)

        if orig_sr != self.cfg.sr:
            wav = torchaudio.functional.resample(wav, orig_sr, self.cfg.sr)

        wav = wav.mean(dim=0, keepdim=True)  # mono (1, N)
        wav = wav / (wav.abs().max() + 1e-6)  # normalize
        return wav

    def to_log_mel(self, wav: torch.Tensor) -> torch.Tensor:
        mel = self.mel(wav)                 # (1, n_mels, T)
        mel = torch.log(mel + 1e-6)
        mel = mel.squeeze(0).transpose(0, 1).contiguous()  # (T, n_mels)
        return mel

    def crop_or_pad(self, wav: torch.Tensor, target_samples: int) -> torch.Tensor:
        n = wav.shape[-1]
        if n > target_samples:
            start = random.randint(0, n - target_samples)
            return wav[..., start:start + target_samples]
        elif n < target_samples:
            pad = target_samples - n
            return torch.nn.functional.pad(wav, (0, pad))
        else:
            return wav


# =========================================================
# log-mel → spike (ΔE ON/OFF)
# =========================================================
def logmel_to_spikes(logmel: torch.Tensor, cfg: AudioSpikeConfig) -> torch.Tensor:
    """
    logmel: (T, n_mels)
    return: (T, 2 * n_mels)
    """
    delta = logmel[1:] - logmel[:-1]
    delta = torch.cat([torch.zeros_like(delta[:1]), delta], dim=0)

    on_spike = (delta > cfg.theta_on).float()
    off_spike = (delta < cfg.theta_off).float()

    spikes = torch.cat([on_spike, off_spike], dim=1)
    return spikes


# =========================================================
# Build Baroque → Romantic waveform
# =========================================================
def build_style_change_waveform(
    wav_a: torch.Tensor,
    wav_b: torch.Tensor,
    cfg: AudioSpikeConfig,
) -> Tuple[torch.Tensor, int]:

    total_samples = int(cfg.clip_seconds * cfg.sr)

    if cfg.change_seconds is None:
        change_seconds = cfg.clip_seconds * random.uniform(0.30, 0.70)
    else:
        change_seconds = cfg.change_seconds

    change_samples = int(change_seconds * cfg.sr)
    change_samples = max(1, min(change_samples, total_samples - 1))

    wav_a = wav_a[..., :change_samples]
    wav_b = wav_b[..., :total_samples - change_samples]

    if wav_a.shape[-1] < change_samples:
        wav_a = torch.nn.functional.pad(wav_a, (0, change_samples - wav_a.shape[-1]))
    if wav_b.shape[-1] < total_samples - change_samples:
        wav_b = torch.nn.functional.pad(
            wav_b,
            (0, (total_samples - change_samples) - wav_b.shape[-1]),
        )

    wav_mix = torch.cat([wav_a, wav_b], dim=-1)

    change_frame = int(change_samples / cfg.hop_length)
    return wav_mix, change_frame


def make_segmentation_labels(T: int, change_frame: int, device: torch.device) -> torch.Tensor:
    y = torch.zeros(T, dtype=torch.long, device=device)
    change_frame = max(0, min(change_frame, T))
    y[change_frame:] = 1
    return y


# =========================================================
# Main batcher (FULL SEQUENCE)
# =========================================================
class AudioStyleChangeBatcher:
    """
    FULL-sequence audio task:
    - Input: 10s spike sequence  (T ≈ 1000)
    - Target: frame-wise style label
    """

    def __init__(self, cfg: AudioSpikeConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

        if cfg.baroque_dir is None or cfg.romantic_dir is None:
            cfg.baroque_dir, cfg.romantic_dir = _default_data_dirs()

        self.baroque_files = _list_audio_files(cfg.baroque_dir)
        self.romantic_files = _list_audio_files(cfg.romantic_dir)

        if len(self.baroque_files) == 0 or len(self.romantic_files) == 0:
            raise RuntimeError(
                "Audio files not found.\n"
                f"{cfg.baroque_dir}\n{cfg.romantic_dir}"
            )

        # self.frontend = AudioToLogMel(cfg, device)
        self.frontend = AudioToLogMel(cfg, device=torch.device("cpu"))


    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        B = self.cfg.batch_size
        target_samples = int(self.cfg.clip_seconds * self.cfg.sr)

        spikes_list = []
        labels_list = []

        for _ in range(B):
            a_path = random.choice(self.baroque_files)
            b_path = random.choice(self.romantic_files)

            wav_a = self.frontend.load_mono_resample(a_path)
            wav_b = self.frontend.load_mono_resample(b_path)

            wav_a = self.frontend.crop_or_pad(wav_a, target_samples)
            wav_b = self.frontend.crop_or_pad(wav_b, target_samples)

            wav_mix, change_frame = build_style_change_waveform(wav_a, wav_b, self.cfg)

            logmel = self.frontend.to_log_mel(wav_mix)
            spikes = logmel_to_spikes(logmel, self.cfg)

            T = spikes.shape[0]
            labels = make_segmentation_labels(T, change_frame, self.device)

            spikes_list.append(spikes)
            labels_list.append(labels)

        maxT = max(s.shape[0] for s in spikes_list)
        input_dim = spikes_list[0].shape[1]

        values = torch.zeros(B, maxT, input_dim)
        targets = torch.zeros(B, maxT, dtype=torch.long)

        for i, (s, y) in enumerate(zip(spikes_list, labels_list)):
            t = s.shape[0]
            values[i, :t] = s
            targets[i, :t] = y

        return values.to(self.device), targets.to(self.device)
