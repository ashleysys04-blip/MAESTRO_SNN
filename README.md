# repo 주소랑 가상환경
conda activate dh_snn
clone한 dh-snn 레포 : https://github.com/eva1801/DH-SNN
내 레포 : https://github.com/ashleysys04-blip/MAESTRO_SNN


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

## What is implemented so far (code tour)
이 섹션은 지금까지 추가/수정된 코드의 역할을 요약합니다.

### 1) Data parsing + split
- `datasets/maestro_metadata.py`
  - MAESTRO CSV를 읽어 `piece_id`와 `performance_id`를 정의합니다.
  - performance-disjoint split을 위해 piece 기반 grouping을 제공합니다.
- `scripts/build_splits.py`
  - `data/maestro/metadata/maestro-v3.0.0.csv`를 읽고
    performance-disjoint `train/val/test` JSON을 생성합니다.
  - 생성 결과: `data/maestro/splits/{train,val,test}.json`

### 2) MIDI 기반 분석 (Figure 3/4용)
- `datasets/maestro_midi.py`
  - MIDI에서 프레임 단위 polyphony k(t)와 onset o(t)을 추출합니다.
  - `mono / chordal / arpeggiated` 라벨링 로직 포함.
- `scripts/extract_polyphony.py`
  - split에 포함된 모든 performance에 대해 위 특징을 계산하고
    `analysis/polyphony_labels.csv`로 저장합니다.

### 3) DH-SNN 통합 + branch 로깅
- `models/dh_snn_wrapper.py`
  - `add_dhsnn_to_path()`로 DH-SNN 경로를 자동 등록합니다.
  - `resolve_snn_layers_root()`로 `SNN_layers` 위치를 찾아 `sys.path`에 추가합니다.
  - `ActivationRecorder`로 branch 관련 텐서(`d_input` 등)를 hook하여 수집합니다.
  - `DHSNNWrapper`는 forward 시 embedding과 branch 기록을 함께 반환합니다.

### 4) 점검 스크립트
- `scripts/inspect_dhsnn.py`
  - DH-SNN 경로 및 `SNN_layers` import 가능 여부를 검사합니다.
  - 사용 예: `python -m scripts.inspect_dhsnn`

### 5) 분석/실험/설정 스켈레톤
- `analysis/`: Fig 3/4를 위한 분석 스크립트 자리 (`branch_activation.py`, `psd_analysis.py`, `plot_utils.py`)
- `experiments/`: 학습/평가 스크립트 자리 (`train_retrieval.py`, `eval_retrieval.py`, `sanity_check.py`)
- `configs/`: 모델/학습/데이터 설정 YAML 자리 (`model.yaml`, `training.yaml`, `data.yaml`)

## Data notes
- MAESTRO CSV 위치: `data/maestro/metadata/maestro-v3.0.0.csv`
- MIDI는 `data/maestro/midi/` 아래에 위치해야 합니다.
- `data/maestro/midi/`는 Git에서 제외되며 로컬에만 유지됩니다.

## DH-SNN import issues
DH-SNN은 Python 패키지로 배포되지 않아서 `pip install -e`가 안됩니다.
따라서 `models/dh_snn_wrapper.py`의 경로 추가 로직을 통해 import를 해결합니다.
