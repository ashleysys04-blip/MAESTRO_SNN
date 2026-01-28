"""
Extract MIDI-based polyphony / onset features for all performances
in train/val/test splits.

This script produces a CSV table used for:
- Figure 3: polyphony-condition vs branch activation
- Analysis grouping (mono / chordal / arpeggiated)

IMPORTANT:
- MIDI is used ONLY for analysis/annotation
- Models are trained and evaluated on audio only
"""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from datasets.maestro_metadata import MaestroMetadata
from datasets.maestro_midi import (
    PolyphonyConfig,
    extract_frame_features,
    label_polyphony_condition,
    summarize_features,
)


# -----------------------------
# Configuration
# -----------------------------

HOP_SEC = 0.02
MIN_DURATION_SEC = 3.0

# Optional: truncate MIDI analysis to first N seconds (None = full length)
MAX_ANALYSIS_DURATION = None  # e.g., 60.0 for faster debugging


# -----------------------------
# Main
# -----------------------------

def load_split_ids(split_path: Path) -> List[str]:
    with open(split_path, "r") as f:
        return json.load(f)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data" / "maestro"

    csv_path = data_root / "metadata" / "maestro-v3.0.0.csv"
    splits_dir = data_root / "splits"
    output_dir = repo_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata = MaestroMetadata(
        csv_path=csv_path,
        maestro_root=data_root,
    )

    # Build lookup: performance_id -> MaestroPerformance
    perf_by_id = {p.performance_id: p for p in metadata.performances}

    # Load splits
    split_names = ["train", "val", "test"]
    split_ids: Dict[str, List[str]] = {
        name: load_split_ids(splits_dir / f"{name}.json")
        for name in split_names
    }

    cfg = PolyphonyConfig(
        hop_sec=HOP_SEC,
        min_duration_sec=MIN_DURATION_SEC,
    )

    rows = []

    # -----------------------------
    # Iterate all splits
    # -----------------------------

    for split_name, perf_ids in split_ids.items():
        print(f"\nProcessing split: {split_name} ({len(perf_ids)} performances)")

        for pid in perf_ids:
            perf = perf_by_id.get(pid)
            if perf is None:
                print(f"[WARN] performance_id not found in metadata: {pid}")
                continue

            midi_path = perf.midi_path
            if not midi_path.exists():
                print(f"[WARN] MIDI file not found: {midi_path}")
                continue

            try:
                feats = extract_frame_features(
                    midi_path,
                    cfg,
                    end_time=MAX_ANALYSIS_DURATION,
                )
                label = label_polyphony_condition(feats, cfg)
                summary = summarize_features(feats, cfg)

            except Exception as e:
                print(f"[ERROR] Failed processing {midi_path}: {e}")
                continue

            row = {
                "split": split_name,
                "piece_id": perf.piece_id,
                "performance_id": perf.performance_id,
                "year": perf.year,
                "midi_path": str(midi_path),
                "polyphony_label": label,
            }
            row.update(summary)
            rows.append(row)

    # -----------------------------
    # Save CSV
    # -----------------------------

    df = pd.DataFrame(rows)
    out_csv = output_dir / "polyphony_labels.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== Extraction complete ===")
    print(f"Saved to: {out_csv}")
    print("\nLabel distribution:")
    print(df["polyphony_label"].value_counts())
    print("\nSplit distribution:")
    print(df.groupby(["split", "polyphony_label"]).size())


if __name__ == "__main__":
    main()
