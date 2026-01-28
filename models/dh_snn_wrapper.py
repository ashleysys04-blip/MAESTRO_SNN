# models/dh_snn_wrapper.py



from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Callable

import torch
import torch.nn as nn


def resolve_snn_layers_root(
    dhsnn_path: Path,
    prefer_subdir: Optional[str] = None,
) -> Optional[Path]:
    """
    Find a directory that contains the SNN_layers package and return its parent.
    """
    candidates: List[Path] = []

    if prefer_subdir:
        preferred = dhsnn_path / prefer_subdir
        if (preferred / "SNN_layers").is_dir():
            candidates.append(preferred)

    if not candidates:
        priority = [
            "TIMIT",
            "SHD",
            "SSC",
            "GSC",
            "s-mnist",
            "NeuroVPR",
            "deap",
            "delayed_xor",
            "multitimescale_xor",
        ]
        for name in priority:
            p = dhsnn_path / name
            if (p / "SNN_layers").is_dir():
                candidates.append(p)

    if not candidates:
        for p in dhsnn_path.rglob("SNN_layers"):
            if p.is_dir():
                candidates.append(p.parent)

    return candidates[0] if candidates else None


def add_dhsnn_to_path(
    repo_root: Optional[Path] = None,
    prefer_subdir: Optional[str] = None,
) -> Path:
    """
    Ensure external/DH-SNN is importable.
    Returns the resolved DH-SNN path.
    """
    import sys

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]

    dhsnn_path = (repo_root / "external" / "DH-SNN").resolve()
    if not dhsnn_path.exists():
        raise FileNotFoundError(
            f"DH-SNN repo not found at {dhsnn_path}\n"
            f"Expected: external/DH-SNN (git submodule or clone)."
        )

    # Put DH-SNN at front so project-level imports resolve
    if str(dhsnn_path) not in sys.path:
        sys.path.insert(0, str(dhsnn_path))

    # Add the parent of SNN_layers so `import SNN_layers` works.
    snn_root = resolve_snn_layers_root(dhsnn_path, prefer_subdir=prefer_subdir)
    if snn_root is not None and str(snn_root) not in sys.path:
        sys.path.insert(0, str(snn_root))

    return dhsnn_path


@dataclass
class BranchRecord:
    """
    Container for recorded branch-related activations.
    Tensors are stored on CPU to avoid GPU memory blow-up.
    """
    # key -> tensor (typically [T, N, B] or [B, ...])
    tensors: Dict[str, torch.Tensor]


class ActivationRecorder:
    """
    Register forward hooks on modules that *likely* correspond to dendritic/denri layers,
    and store any tensors that look like branch-wise activations.

    You can refine the selection rules once you see DH-SNN's actual module names.
    """
    def __init__(
        self,
        keep: bool = True,
        to_cpu: bool = True,
        max_items: int = 64,
        name_filter: Optional[Callable[[str], bool]] = None,
    ):
        self.keep = keep
        self.to_cpu = to_cpu
        self.max_items = max_items
        self.name_filter = name_filter
        self._hooks: List[Any] = []
        self._records: Dict[str, torch.Tensor] = {}

    def clear(self):
        self._records = {}

    def records(self) -> BranchRecord:
        return BranchRecord(tensors=dict(self._records))

    def close(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []

    def _maybe_store(self, key: str, x: Any):
        if not self.keep:
            return
        if len(self._records) >= self.max_items:
            return

        if torch.is_tensor(x):
            t = x.detach()
            if self.to_cpu:
                t = t.to("cpu")
            self._records[key] = t

    def _hook_fn(self, module_name: str):
        def fn(module: nn.Module, inputs: Tuple[Any, ...], output: Any):
            # 1) store module attributes if exist
            for attr in ["d_input", "denri_input", "branch_input", "I_d", "i_d", "dend_current"]:
                if hasattr(module, attr):
                    val = getattr(module, attr)
                    if torch.is_tensor(val):
                        self._maybe_store(f"{module_name}.{attr}", val)

            # 2) store output if it looks like a branch tensor
            # output can be Tensor, tuple, dict...
            def scan(obj: Any, prefix: str):
                if torch.is_tensor(obj):
                    # heuristic: branch tensors often have >=3 dims or include 'branch' in prefix
                    if obj.dim() >= 3 or ("branch" in prefix) or ("denri" in prefix) or ("dend" in prefix):
                        self._maybe_store(prefix, obj)
                elif isinstance(obj, dict):
                    for k, v in obj.items():
                        scan(v, f"{prefix}.{k}")
                elif isinstance(obj, (tuple, list)):
                    for i, v in enumerate(obj):
                        scan(v, f"{prefix}[{i}]")

            scan(output, f"{module_name}.out")
        return fn

    def attach(self, model: nn.Module):
        """
        Attach hooks to selected modules.
        """
        self.close()
        self.clear()

        for name, m in model.named_modules():
            lname = name.lower()

            # Default heuristic: modules whose name indicates dendritic/denri
            select = ("denri" in lname) or ("dend" in lname) or ("branch" in lname)
            if self.name_filter is not None:
                select = select and self.name_filter(name)

            if select:
                self._hooks.append(m.register_forward_hook(self._hook_fn(name)))


class DHSNNWrapper(nn.Module):
    """
    Wrap a DH-SNN backbone and optionally record branch activations during forward.

    backbone: any nn.Module
      - forward(x, ...) -> embedding or logits
    """
    def __init__(self, backbone: nn.Module, recorder: Optional[ActivationRecorder] = None):
        super().__init__()
        self.backbone = backbone
        self.recorder = recorder

        if self.recorder is not None:
            self.recorder.attach(self.backbone)

    def forward(self, x: torch.Tensor, **kwargs):
        """
        Returns:
          y: backbone output
          record: BranchRecord or None
        """
        if self.recorder is not None:
            self.recorder.clear()

        y = self.backbone(x, **kwargs)

        rec = self.recorder.records() if self.recorder is not None else None
        return y, rec
