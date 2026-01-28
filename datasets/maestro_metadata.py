"""
MAESTRO metadata parsing utilities.

This module defines:
- What is a "piece"
- What is a "performance"
- How MAESTRO files are grouped for performance-disjoint splits

Design philosophy:
- A musical piece may have multiple performances (years/recordings)
- Splits should be disjoint at the performance level, not piece level
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import pandas as pd


@dataclass
class MaestroPerformance:
    """
    One concrete performance/recording of a musical piece.
    """
    piece_id: str
    performance_id: str
    year: int
    midi_path: Path
    audio_path: Path
    duration: float
    split: str  # original MAESTRO split (train/validation/test)


class MaestroMetadata:
    """
    Wrapper around MAESTRO metadata CSV.

    Responsible for:
    - Parsing CSV
    - Defining piece_id and performance_id
    - Grouping performances by piece
    """

    def __init__(
        self,
        csv_path: Path,
        maestro_root: Path,
    ):
        """
        Args:
            csv_path: path to maestro-v3.0.0.csv
            maestro_root: root directory containing midi/ and audio/
        """
        self.csv_path = Path(csv_path)
        self.maestro_root = Path(maestro_root)

        self.df = pd.read_csv(self.csv_path)
        self.performances: List[MaestroPerformance] = []

        self._parse_metadata()

    # ------------------------------------------------------------------
    # Core parsing logic
    # ------------------------------------------------------------------

    def _parse_metadata(self):
        """
        Parse MAESTRO CSV into MaestroPerformance objects.
        """
        for _, row in self.df.iterrows():
            piece_id = self._make_piece_id(row)
            performance_id = self._make_performance_id(row)

            midi_path = self.maestro_root / row["midi_filename"]
            audio_path = self.maestro_root / row["audio_filename"]

            perf = MaestroPerformance(
                piece_id=piece_id,
                performance_id=performance_id,
                year=int(row["year"]),
                midi_path=midi_path,
                audio_path=audio_path,
                duration=float(row["duration"]),
                split=row["split"],
            )
            self.performances.append(perf)

    # ------------------------------------------------------------------
    # ID definitions (논문에서 가장 중요한 부분)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_piece_id(row) -> str:
        """
        Define a unique identifier for a musical piece.

        We use (composer, title) as piece identity.
        This matches the musical notion of "same piece, different performance".

        Example:
            piece_id = "Chopin__Nocturne_Op9_No2"
        """
        composer = row["canonical_composer"].replace(" ", "_")
        title = row["canonical_title"].replace(" ", "_")
        return f"{composer}__{title}"

    @staticmethod
    def _make_performance_id(row) -> str:
        """
        Define a unique identifier for a performance.

        Performance = (piece_id, year, MIDI filename)
        """
        year = row["year"]
        midi_name = Path(row["midi_filename"]).stem
        return f"{year}__{midi_name}"

    # ------------------------------------------------------------------
    # Grouping utilities
    # ------------------------------------------------------------------

    def performances_by_piece(self) -> Dict[str, List[MaestroPerformance]]:
        """
        Group performances by piece_id.
        """
        groups: Dict[str, List[MaestroPerformance]] = {}
        for p in self.performances:
            groups.setdefault(p.piece_id, []).append(p)
        return groups

    def list_pieces(self) -> List[str]:
        """
        List all unique piece IDs.
        """
        return sorted({p.piece_id for p in self.performances})

    def list_performances(self) -> List[str]:
        """
        List all unique performance IDs.
        """
        return sorted({p.performance_id for p in self.performances})

    # ------------------------------------------------------------------
    # Convenience filters
    # ------------------------------------------------------------------

    def filter_by_min_duration(self, min_duration: float):
        """
        Remove performances shorter than a threshold (in seconds).
        Useful for ensuring enough clips can be sampled.
        """
        self.performances = [
            p for p in self.performances if p.duration >= min_duration
        ]

    def filter_pieces_with_multiple_performances(self, min_performances: int = 2):
        """
        Keep only pieces with >= min_performances performances.
        Critical for performance-disjoint evaluation.
        """
        groups = self.performances_by_piece()
        allowed_pieces = {
            piece for piece, perfs in groups.items()
            if len(perfs) >= min_performances
        }
        self.performances = [
            p for p in self.performances if p.piece_id in allowed_pieces
        ]
