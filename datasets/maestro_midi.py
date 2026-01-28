"""

실행 방법
python -m datasets.maestro_midi \
  --midi "/Users/ashleyseo/Desktop/KAIST/9. KAIST 2025 겨울/MAESTRO_SNN/data/maestro/midi/maestro-v3.0.0/2018/MIDI-Unprocessed_Recital1-3_MID--AUDIO_03_R1_2018_wav--1.midi" \
  --hop 0.02

(파일은 경로 바꾸면서 바꾸면 됨)


MIDI-based polyphony / onset features for MAESTRO.

We use MAESTRO's aligned MIDI to derive:
- k(t): number of active notes per frame
- o(t): onset indicator per frame
- rho_o: onset density
- labels: mono / chordal / arpeggiated based on operational definitions

This file is designed for:
- Figure 3: polyphony-condition branch specialization
- Figure 4: onset/sustain (fast vs slow) time-scale analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import pretty_midi
except ImportError as e:
    raise ImportError(
        "pretty_midi is required. Install with: pip install pretty_midi mido\n"
        "If you see numpy compatibility issues, try: pip install 'numpy<2' --force-reinstall"
    ) from e


@dataclass
class PolyphonyConfig:
    """
    Operational definition thresholds.

    Notes:
    - hop_sec must match your audio feature hop if you want perfect alignment.
      For MIDI-only analysis, any reasonable hop (e.g., 0.02s) works.
    """
    hop_sec: float = 0.02  # 20 ms
    vel_threshold: int = 1  # ignore velocity=0 as note-off; keep >=1
    # Label thresholds
    mono_mean_k_max: float = 1.2
    poly_mean_k_min: float = 2.0
    onset_density_threshold: float = 0.08  # ~8 onsets/sec if hop=0.02? (see note below)
    # Minimum duration to consider (seconds)
    min_duration_sec: float = 3.0


@dataclass
class MidiFrameFeatures:
    """
    Frame-wise features extracted from MIDI.
    """
    times: np.ndarray            # shape [T], frame start times
    k_active: np.ndarray         # shape [T], active note count
    onset: np.ndarray            # shape [T], onset indicator (0/1)
    mean_k: float                # average polyphony
    onset_density: float         # fraction of frames with onset (NOT per-second)
    duration_sec: float          # clip duration in seconds


def _frame_times(duration: float, hop: float) -> np.ndarray:
    if duration <= 0:
        return np.zeros((0,), dtype=np.float32)
    T = int(np.ceil(duration / hop))
    return (np.arange(T, dtype=np.float32) * hop)


def extract_frame_features(
    midi_path: Path,
    cfg: PolyphonyConfig,
    end_time: Optional[float] = None,
) -> MidiFrameFeatures:
    """
    Extract k(t) and onset o(t) from a MIDI file.

    Args:
        midi_path: path to .mid
        cfg: PolyphonyConfig
        end_time: optionally truncate analysis to end_time seconds.

    Returns:
        MidiFrameFeatures
    """
    midi_path = Path(midi_path)
    pm = pretty_midi.PrettyMIDI(str(midi_path))

    # Determine duration
    duration = pm.get_end_time()
    if end_time is not None:
        duration = min(duration, float(end_time))

    if duration < cfg.min_duration_sec:
        times = _frame_times(duration, cfg.hop_sec)
        return MidiFrameFeatures(
            times=times,
            k_active=np.zeros_like(times, dtype=np.int32),
            onset=np.zeros_like(times, dtype=np.int32),
            mean_k=0.0,
            onset_density=0.0,
            duration_sec=duration,
        )

    times = _frame_times(duration, cfg.hop_sec)
    T = len(times)

    # Active-note counting:
    # We'll create two event arrays:
    #  - note_on_events[t] = number of note-on events occurring in frame t
    #  - active_count[t] = active note count in frame t (after applying events)
    note_on_events = np.zeros((T,), dtype=np.int32)
    note_delta = np.zeros((T + 1,), dtype=np.int32)  # delta encoding for active notes

    # Iterate all notes across all instruments
    for inst in pm.instruments:
        for note in inst.notes:
            # pretty_midi gives note.start, note.end, note.velocity
            if note.velocity < cfg.vel_threshold:
                continue
            if note.start >= duration:
                continue

            start = max(0.0, float(note.start))
            end = min(duration, float(note.end))
            if end <= 0.0 or end <= start:
                continue

            t_on = int(start / cfg.hop_sec)
            t_off = int(end / cfg.hop_sec)

            # Onset event
            if 0 <= t_on < T:
                note_on_events[t_on] += 1

            # Active note delta: +1 at t_on, -1 at t_off
            t_on = max(0, min(t_on, T))
            t_off = max(0, min(t_off, T))
            note_delta[t_on] += 1
            note_delta[t_off] -= 1

    # Convert delta to active counts
    k_active = np.cumsum(note_delta[:-1]).astype(np.int32)

    # Onset indicator (binary): 1 if any note-on occurs at that frame
    onset = (note_on_events > 0).astype(np.int32)

    mean_k = float(np.mean(k_active)) if T > 0 else 0.0
    onset_density = float(np.mean(onset)) if T > 0 else 0.0  # fraction of frames

    return MidiFrameFeatures(
        times=times,
        k_active=k_active,
        onset=onset,
        mean_k=mean_k,
        onset_density=onset_density,
        duration_sec=duration,
    )


def label_polyphony_condition(
    feats: MidiFrameFeatures,
    cfg: PolyphonyConfig,
) -> str:
    """
    Label a segment as one of:
        - "mono"
        - "chordal"
        - "arpeggiated"
        - "other"

    Rules (operational definitions):
      mono: mean_k <= mono_mean_k_max
      chordal: mean_k >= poly_mean_k_min AND onset_density <= onset_density_threshold
      arpeggiated: mean_k >= poly_mean_k_min AND onset_density > onset_density_threshold

    Notes:
    - onset_density here is "fraction of frames with any onset".
      If hop_sec=0.02, then onset_density=0.1 roughly means onsets in 10% of frames.
      You can also convert to per-second: onset_rate_hz = onset_density / hop_sec.
    """
    if feats.duration_sec < cfg.min_duration_sec:
        return "other"

    if feats.mean_k <= cfg.mono_mean_k_max:
        return "mono"

    if feats.mean_k >= cfg.poly_mean_k_min:
        if feats.onset_density <= cfg.onset_density_threshold:
            return "chordal"
        else:
            return "arpeggiated"

    return "other"


def summarize_features(feats: MidiFrameFeatures, cfg: PolyphonyConfig) -> Dict[str, float]:
    """
    Convenient scalar summary (for logging / CSV).
    """
    onset_rate_hz = (feats.onset_density / cfg.hop_sec) if cfg.hop_sec > 0 else 0.0
    return {
        "duration_sec": float(feats.duration_sec),
        "mean_k": float(feats.mean_k),
        "onset_density_frame_frac": float(feats.onset_density),
        "onset_rate_hz": float(onset_rate_hz),
    }


# ---------------------------
# Quick sanity check CLI
# ---------------------------
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--midi", type=str, required=True, help="Path to a MIDI file")
    parser.add_argument("--hop", type=float, default=0.02)
    parser.add_argument("--end", type=float, default=None, help="Optional truncation end time (sec)")
    args = parser.parse_args()

    cfg = PolyphonyConfig(hop_sec=args.hop)
    feats = extract_frame_features(Path(args.midi), cfg, end_time=args.end)
    label = label_polyphony_condition(feats, cfg)

    out = summarize_features(feats, cfg)
    out["label"] = label

    print(json.dumps(out, indent=2))
