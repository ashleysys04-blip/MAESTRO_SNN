# Spiking Neural Networks with Neuron-level Multi-timescale Dynamics for Music Recognition

## Project Overview

This repository investigates whether neuron-level multi-timescale dynamics improve music recognition performance when a conventional CNN architecture is converted into a spiking neural network (SNN).

The core idea of this project is to **preserve the original CNN topology** used in prior music recognition work while **varying only the neuron dynamics**. By fixing the network structure and task definition and changing only the temporal properties of spiking neurons, this work enables a controlled comparison between single-timescale and multi-timescale spiking neural models.

---

# repo 주소랑 가상환경
conda activate dh_snn
clone한 dh-snn 레포 : https://github.com/eva1801/DH-SNN
내 레포 : https://github.com/ashleysys04-blip/MAESTRO_SNN


# MAESTRO_SNN

Dendritic multi-timescale spiking neural networks for
short-clip identification of classical music.

https://magenta.withgoogle.com/datasets/maestro#v300


## Overview

MAESTRO_SNN/
│
├── data/ maestro
│
├── experiments/
│   ├── lif_baseline/
│   ├── dh_all_layers/
│
├── external 이랑 model 은 DH-SNN 실험용
│
├── models/
│   ├── cnn_reference.py
│   ├── spiking_cnn_lif.py
│   ├── spiking_cnn_dh_lif.py
│   ├── neurons/
│   │   ├── lif.py
│   │   └── dh_lif.py
│   └── layers/
│       ├── spiking_conv.py
│       └── readout.py
│
└── README.md


## Task Description

The task setting follows a prior CNN-based music recognition study. [1]

### Input

Music signals are segmented into fixed-length excerpts and converted into acoustic features:

- MFCC
- Chroma
- ST-RMS

Noise augmentation is applied during training and evaluation to test robustness.

### Output

The network performs multi-head classification with three output heads:

- Title classification
- Style classification
- Emotion classification

Each head is trained with a supervised classification loss, and the total loss is defined as a weighted sum of the individual head losses.

---

## Fixed Network Structure

Across all experiments, the **network topology is kept fixed**.

The following components are identical for all models:

- CNN architecture (number of layers and connectivity)
- Convolution and pooling structure
- Feature aggregation strategy
- Multi-head classifier design
- Training procedure and optimization method

This ensures that performance differences are attributable solely to differences in neuron dynamics rather than architectural changes.

---

## Models Compared

Two spiking neural network models are compared.

### Model A: Single-timescale Spiking CNN (Baseline)

This model is obtained by directly converting the original CNN into a spiking neural network.

- Uses standard leaky integrate-and-fire (LIF) neurons
- Each neuron has a single membrane time constant
- Temporal processing relies on spike accumulation and readout integration
- No explicit mechanism for handling multiple temporal scales inside individual neurons

This model serves as a baseline spiking implementation of the CNN.

---

### Model B: Multi-timescale Spiking CNN (Proposed)

This model uses **exactly the same network topology** as Model A.

The only difference is the neuron model used in all layers.

- Each neuron contains multiple dendritic branches
- Each dendritic branch has its own time constant
- Different branches capture fast and slow temporal components simultaneously
- Dendritic branches are sparsely or disjointly connected to input channels to avoid parameter explosion
- Branch-level timing factors are learnable

By introducing neuron-level multi-timescale dynamics in **every layer**, this model enables hierarchical temporal feature extraction without altering the network graph.

---

## Experimental Comparison Strategy

The comparison between Model A and Model B is strictly controlled:

- Same task
- Same data
- Same network topology
- Same loss functions
- Same optimization method

The only difference is the presence or absence of neuron-level multi-timescale dynamics.

This allows direct evaluation of how dendritic heterogeneity affects music recognition accuracy, emotion classification performance, and robustness to noise.

Optional ablation studies include variants where multi-timescale neurons are applied only to specific parts of the network (e.g., early layers or late layers), but the primary comparison focuses on the all-layer setting.

---

## Relation to Prior Work

This project integrates ideas from three complementary lines of research.

### CNN-based Music Recognition [1]

A prior CNN-based music recognition study provides:

- Task definition
- Input feature design (MFCC, Chroma, ST-RMS)
- CNN architecture
- Multi-head classification setup

These components are reused without modification to establish a strong and fair baseline.

---

### Biologically Inspired Spiking Neural Networks for Sequential Learning [2]

A biologically inspired spiking neural network study motivates the importance of temporal structure in music and sequence processing.

Key ideas include:

- Temporal information should not be treated solely as a hidden state
- Fast and slow temporal components benefit from separate representations
- Spiking neural networks are naturally suited for temporal-sequential tasks

This work provides the conceptual motivation for introducing temporal inductive bias into music recognition models.

---

### Temporal Dendritic Heterogeneity in Spiking Neural Networks [3]

A study on temporal dendritic heterogeneity provides the mechanistic foundation for the proposed neuron model.

Key ideas adopted include:

- Multi-timescale temporal memory can be implemented at the neuron level
- Dendritic branches with different time constants can simultaneously capture short-term and long-term dynamics
- Neuron dynamics and network topology can be decoupled

These ideas are directly applied to construct the multi-timescale spiking neurons used in Model B.

---

## Research Question

This repository addresses the following research question:

**Can neuron-level multi-timescale dynamics improve music recognition performance in spiking neural networks when the overall CNN structure is preserved?**

By isolating neuron dynamics as the only variable, this project provides a principled evaluation of temporal modeling at the neuron level for music understanding tasks.


[1] Y. Shi, "A CNN-Based Approach for Classical Music Recognition and Style Emotion Classification," in IEEE Access, vol. 13, pp. 20647-20666, 2025, doi: 10.1109/ACCESS.2025.3535411.
keywords: {Emotion recognition;Accuracy;Feature extraction;Multiple signal classification;Classification algorithms;Noise;Deep learning;Recommender systems;Proposals;Heuristic algorithms;CNN;deep learning;music recognition;music retrieval;optimization algorithm},

[2] Temporal-Sequential Learning With a Brain-Inspired Spiking Neural Network and Its Application to Musical Memory

[3] Zheng, H., Zheng, Z., Hu, R. et al. Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics. Nat Commun 15, 277 (2024). https://doi.org/10.1038/s41467-023-44614-z