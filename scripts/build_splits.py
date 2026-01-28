"""
Build performance-disjoint train/val/test splits for MAESTRO.


실행

python -m scripts.build_splits



Key design:
- Musical pieces are fixed across splits
- Performances (recordings) are disjoint between splits
- Only pieces with >=2 performances are used

This script produces:
- train.json
- val.json
- test.json

Each JSON contains a list of performance_ids.
"""

import json
import random
from pathlib import Path
from typing import Dict, List

from datasets.maestro_metadata import MaestroMetadata, MaestroPerformance


# -----------------------------
# Configuration
# -----------------------------

RANDOM_SEED = 42

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

MIN_PERFORMANCES_PER_PIECE = 2
MIN_DURATION = 60.0  # seconds (ensure enough clips)


# -----------------------------
# Split logic
# -----------------------------

def split_performances(
    performances: List[MaestroPerformance],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
):
    """
    Split a list of performances into train/val/test.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    n = len(performances)
    random.shuffle(performances)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = performances[:n_train]
    val = performances[n_train:n_train + n_val]
    test = performances[n_train + n_val:]

    # Edge case: if too few performances, ensure test is non-empty
    if len(test) == 0 and len(train) > 1:
        test.append(train.pop())

    return train, val, test


# -----------------------------
# Main
# -----------------------------

def main():
    random.seed(RANDOM_SEED)

    # Paths (adjust if needed)
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data" / "maestro"
    csv_path = data_root / "metadata" / "maestro-v3.0.0.csv"
    splits_dir = data_root / "splits"

    splits_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata = MaestroMetadata(
        csv_path=csv_path,
        maestro_root=data_root,
    )

    # Basic filtering
    metadata.filter_by_min_duration(MIN_DURATION)
    metadata.filter_pieces_with_multiple_performances(
        MIN_PERFORMANCES_PER_PIECE
    )

    pieces = metadata.performances_by_piece()

    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []

    # Split per piece
    for piece_id, perfs in pieces.items():
        if len(perfs) < MIN_PERFORMANCES_PER_PIECE:
            continue

        train, val, test = split_performances(
            perfs,
            TRAIN_RATIO,
            VAL_RATIO,
            TEST_RATIO,
        )

        train_ids.extend([p.performance_id for p in train])
        val_ids.extend([p.performance_id for p in val])
        test_ids.extend([p.performance_id for p in test])

    # Save splits
    with open(splits_dir / "train.json", "w") as f:
        json.dump(sorted(train_ids), f, indent=2)

    with open(splits_dir / "val.json", "w") as f:
        json.dump(sorted(val_ids), f, indent=2)

    with open(splits_dir / "test.json", "w") as f:
        json.dump(sorted(test_ids), f, indent=2)

    # Summary (콘솔 로그는 디버깅 + 논문 sanity check에 유용)
    print("=== Split summary ===")
    print(f"Train performances: {len(train_ids)}")
    print(f"Val performances:   {len(val_ids)}")
    print(f"Test performances:  {len(test_ids)}")
    print(f"Total pieces:       {len(pieces)}")


if __name__ == "__main__":
    main()
