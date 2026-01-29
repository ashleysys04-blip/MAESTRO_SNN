"""
Multi-timescale AUDIO style change task
- Baroque -> Romantic (audio-based)
- Uses DH-SFNN (dendritic spiking dense layer)


external/DH-SNN/
 └─ AUDIO/
     ├─ Mel_spectrogram.py
     ├─ multi_snn_forAUDIO.py   ← 이 파일
     └─ data/
         ├─ baroque/*.wav
         └─ romantic/*.wav

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np

# =========================
# AUDIO batcher import
# =========================
from AUDIO.Mel_spectrogram import (
    AudioSpikeConfig,
    AudioStyleChangeBatcher
)

# =========================
# SNN layers
# =========================
from SNN_layers.spike_dense import *
from SNN_layers.spike_neuron import *
from SNN_layers.spike_rnn import *


torch.manual_seed(42)

device = torch.device("cpu")
print("device:", device)

# =========================
# experiment config
# =========================
time_steps = 1000          # 100 frames = 1s
batch_size = 8          # M1 안전값
hidden_dims = 16
output_dim = 2            # baroque / romantic
input_size = 128          # mel64 × (ON+OFF)

epochs = 150
learning_rate = 1e-3
log_interval = 50

# =========================
# AUDIO batcher
# =========================
cfg = AudioSpikeConfig(
    batch_size=batch_size,
    clip_seconds=10.0
)
batcher = AudioStyleChangeBatcher(cfg, device=device)


def get_batch():
    return batcher.get_batch()

# =========================
# Model
# =========================
class Dense_test_1layer(nn.Module):
    """
    DH-SFNN with 1 dendritic dense layer
    """
    def __init__(self, input_size, hidden_dims, output_dim):
        super().__init__()

        self.dense_1 = spike_dense_test_denri_wotanh_R_xor(
            input_size,
            hidden_dims,
            tau_ninitializer='uniform',
            low_n=2,
            high_n=6,
            vth=1,
            dt=1,
            branch=2,
            device=device,
            bias=False
        )

        self.dense_2 = nn.Linear(hidden_dims, output_dim)
        self.criterion = nn.CrossEntropyLoss()

    def init(self):
        self.dense_1.set_neuron_state(batch_size)

    def forward(self, x, target):
        # x: (B, T, input_size)
        B, T, _ = x.shape

        loss_list = []
        correct = 0
        total = 0

        for t in range(T):
            input_t = x[:, t, :]
            _, spike = self.dense_1.forward(input_t)
            logits = self.dense_2(spike)

            # loss per timestep (do NOT accumulate directly)
            loss_t = self.criterion(logits, target[:, t])
            loss_list.append(loss_t)

            pred = logits.argmax(dim=1)
            correct += (pred == target[:, t]).sum()
            total += B

        # safe aggregation
        loss = torch.stack(loss_list).mean()

        return loss, correct, total

# =========================
# Train loop
# =========================
model = Dense_test_1layer(input_size, hidden_dims, output_dim).to(device)

optimizer = torch.optim.Adam(
    [
        {"params": model.dense_1.dense.weight},
        {"params": model.dense_1.tau_m, "lr": learning_rate},
        {"params": model.dense_1.tau_n, "lr": learning_rate},
        {"params": model.dense_2.parameters()},
    ],
    lr=learning_rate
)

scheduler = StepLR(optimizer, step_size=50, gamma=0.1)


def train():
    for epoch in range(epochs):
        model.train()
        

        total_loss = 0
        total_correct = 0
        total_samples = 0

        for i in range(log_interval):
            model.init()
            
            x, y = get_batch()
            optimizer.zero_grad()

            loss, correct, total = model(x, y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 20)
            optimizer.step()

            if i % 5 == 0:
                print(
                    f"[epoch {epoch} | batch {i}] "
                    f"loss={loss.item():.4f}, "
                    f"acc={(correct/total):.3f}"
                )

            total_loss += loss.item()
            total_correct += correct.item()
            total_samples += total

        scheduler.step()

        acc = total_correct / total_samples
        print(
            f"Epoch {epoch:3d} | "
            f"Loss {total_loss / log_interval:.4f} | "
            f"Acc {acc:.3f}"
        )


# =========================
# run
# =========================
if __name__ == "__main__":
    train()
