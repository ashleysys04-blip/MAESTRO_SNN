# MAESTRO_SNN

Dendritic multi-timescale spiking neural networks for
short-clip identification of classical music.

https://magenta.withgoogle.com/datasets/maestro#v300


## Overview
This repository contains experiments for learning piece-level
representations from short MAESTRO clips using DH-SNNs.

Key features:
- Performance-disjoint retrieval task
- MIDI-based polyphony and onset analysis
- Branch-wise interpretability (time-scale decomposition)

## Setup
1. Clone repository
2. Add DH-SNN as submodule (see DH-SNN integration)
3. Download MAESTRO metadata and MIDI
4. Build splits
5. Train retrieval model

## Structure
- datasets/: MAESTRO parsing and labeling
- models/: DH-SNN wrapper
- experiments/: training and evaluation
- analysis/: figure generation (Fig. 3–4)

## DH-SNN integration
We use the upstream DH-SNN codebase as-is and add logging/saving on our side.

Expected location:
```
external/
  DH-SNN/          (git submodule or git clone)
```

Submodule init/update:
```
git submodule update --init --recursive
```

Runtime flow (conceptual):
audio -> DH-SNN -> embedding
and branch-wise internal signals (e.g., per-branch d_input) are captured
and saved in a consistent format for Figure 3/4 analysis.
