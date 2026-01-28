# MAESTRO_SNN

Dendritic multi-timescale spiking neural networks for
short-clip identification of classical music.

## Overview
This repository contains experiments for learning piece-level
representations from short MAESTRO clips using DH-SNNs.

Key features:
- Performance-disjoint retrieval task
- MIDI-based polyphony and onset analysis
- Branch-wise interpretability (time-scale decomposition)

## Setup
1. Clone repository
2. Add DH-SNN as submodule
3. Download MAESTRO metadata and MIDI
4. Build splits
5. Train retrieval model

## Structure
- datasets/: MAESTRO parsing and labeling
- models/: DH-SNN wrapper
- experiments/: training and evaluation
- analysis/: figure generation (Fig. 3–4)
