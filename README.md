# Asymmetric Configuration Mixing for Adapting CLIP Grid Features in Visual Question Answering

Official PyTorch implementation of **ACM-LXMERT**, which adapts a pretrained LXMERT backbone (originally pretrained with Faster R-CNN region features) to newly introduced frozen CLIP RN50×4 grid features during downstream VQA fine-tuning—without additional vision–language pretraining.

## Overview

During fine-tuning, **Asymmetric Configuration Mixing (ACM)** independently samples one of two visual input configurations for each training example using a Bernoulli indicator $z_i \sim \mathrm{Bernoulli}(p)$ (default $p = 0.6$):

- **CLIP-only ($z_i = 1$, probability $p$)**: Masks all 36 region tokens in visual self-attention and cross-modal attention while keeping all 49 projected CLIP $7\times 7$ grid tokens visible.
- **Joint ($z_i = 0$, probability $1 - p$)**: Retains both the 36 region tokens and the 49 CLIP grid tokens.

Both configurations share all model parameters (LXMERT backbone, VQA prediction head, and trainable CLIP grid projection interface) and optimize the same soft-target binary cross-entropy loss. **Standard inference always uses the unmasked Joint configuration.**

## Main Results

| Model | VQA v2 minival (4 seeds) | VQA v2 test-dev | VQA v2 test-std | GQA balanced testdev (3 seeds) |
| :--- | :---: | :---: | :---: | :---: |
| Plain LXMERT | 70.015 ± 0.111 | 72.42\* | 72.54\* | 59.816 ± 0.361 |
| RG-LXMERT (Direct Joint, $p=0$) | 70.264 ± 0.187 | -- | -- | 59.702 ± 0.275 |
| **ACM-LXMERT ($p=0.6$)** | **72.705 ± 0.023** | **75.41** | **75.65** | **60.375 ± 0.165** |

\* Official LXMERT reported single-model scores. Training seeds: `{9595, 2027, 3407, 42}` on VQA v2 minival and `{9595, 2027, 3407}` on GQA balanced testdev; official VQA v2 test-dev/test-std predictions use the 4th-epoch `train+val` checkpoint trained with seed `9595`.

## Repository Structure

```text
ACM-LXMERT/
├── acm_lxmert/
│   ├── assets/                   # Official LXMERT answer table (all_ans.json) & bert_config.json
│   ├── data/                     # VQA v2 and GQA LMDB + CLIP memmap datasets
│   ├── models/                   # OfficialBaselineVQA & ClipGridVQA (RG-LXMERT / ACM-LXMERT)
│   └── utils.py                  # Seeding, BertAdam, soft-target BCE, fixed-derangement swaps
├── configs/                      # Default VQA v2 and GQA hyperparameter JSON configs
├── scripts/
│   ├── prepare_data/             # TSV-to-LMDB region caching & CLIP RN50x4 grid extraction
│   ├── train_vqa2.py             # VQA v2 training (plain / rg / acm / alternate / two_stage)
│   ├── eval_vqa2.py              # VQA v2 evaluation (Joint, CLIP-only, Joint CLIP Swap, test2015)
│   ├── train_gqa.py              # GQA training (plain / rg / acm)
│   ├── eval_gqa.py               # GQA evaluation (Joint, CLIP-only, CLIP/Region Swap)
│   └── evaluate_gqa_predictions.py
└── third_party/lxmert/           # Upstream LXMERT backbone modules (airsplay/lxmert)
```

## Installation

All experiments were conducted with Python 3.10 and PyTorch 2.5.1 (CUDA 12.4) on a single NVIDIA GeForce RTX 3090 (24 GB):

```bash
conda create -n acm_lxmert python=3.10 -y
conda activate acm_lxmert
pip install -r requirements.txt
```

## Data & Pretrained Checkpoints Preparation

1. **Pretrained Weights**:
   - Download the official pretrained LXMERT checkpoint (`model_LXRT.pth`) from [airsplay/lxmert](https://github.com/airsplay/lxmert).
   - Download the official OpenAI CLIP TorchScript checkpoint (`RN50x4.pt`) from [openai/CLIP](https://github.com/openai/CLIP).

2. **Region Features (Faster R-CNN `obj36`)**:
   - Download the official LXMERT VQA v2 / GQA JSON splits, answer vocabularies (`trainval_label2ans.json`, `trainval_ans2label.json`), and Faster R-CNN `obj36` TSV files from [airsplay/lxmert](https://github.com/airsplay/lxmert).
   - Cache the TSV files into memory-safe LMDB stores:
     ```bash
     # VQA v2 (repeat for val2014 and test2015)
     python scripts/prepare_data/cache_vqa2_tsv_to_lmdb.py \
       --tsv /path/to/train2014_obj36.tsv \
       --output /path/to/lmdb/train2014_obj36.lmdb

     # GQA (repeat for gqa_testdev_obj36.tsv)
     python scripts/prepare_data/cache_gqa_tsv_to_lmdb.py \
       --tsv /path/to/vg_gqa_obj36.tsv \
       --output /path/to/lmdb/vg_obj36.lmdb
     ```

3. **Offline CLIP RN50×4 $7\times 7$ Grid Features**:
   - Extract float16 `[N, 49, 2560]` grid features (448×448 bicubic resize, adaptive average pooling to $7\times 7$):
     ```bash
     # VQA v2 (repeat for val2014 and test2015)
     python scripts/prepare_data/extract_clip_grid_vqa2.py \
       --source /path/to/coco2014/train2014 \
       --split train2014 \
       --clip-model /path/to/RN50x4.pt \
       --output /path/to/clip_features/vqa2_rn50x4_grid7

     # GQA
     python scripts/prepare_data/extract_clip_grid_gqa.py \
       --manifest /path/to/gqa_image_manifest.json \
       --images /path/to/gqa/images \
       --model /path/to/RN50x4.pt \
       --output /path/to/clip_features/gqa_rn50x4_grid7
     ```

## Training

### VQA v2 (minival development runs)

```bash
# ACM-LXMERT (p = 0.6, grid = 7, 4 epochs, lr = 5e-5, batch size = 32)
python scripts/train_vqa2.py \
  --config configs/vqa2_acm.json \
  --vqa-data-root /path/to/lxmert/data/vqa \
  --train-lmdb /path/to/lmdb/train2014_obj36.lmdb \
  --val-lmdb /path/to/lmdb/val2014_obj36.lmdb \
  --clip-root /path/to/clip_features/vqa2_rn50x4_grid7 \
  --official-checkpoint /path/to/model_LXRT.pth \
  --seed 9595 \
  --output outputs/vqa2_acm_seed9595

# Direct Joint baseline (RG-LXMERT, p = 0)
python scripts/train_vqa2.py \
  --config configs/vqa2_rg.json \
  --vqa-data-root /path/to/lxmert/data/vqa \
  --train-lmdb /path/to/lmdb/train2014_obj36.lmdb \
  --val-lmdb /path/to/lmdb/val2014_obj36.lmdb \
  --clip-root /path/to/clip_features/vqa2_rn50x4_grid7 \
  --official-checkpoint /path/to/model_LXRT.pth \
  --seed 9595 \
  --output outputs/vqa2_rg_seed9595
```

To train the final VQA v2 model on all labeled `train + nominival + minival` questions (658,111 questions) for official `test-dev` / `test-std` evaluation, pass `--train-splits train nominival minival --no-val`.

### GQA

```bash
python scripts/train_gqa.py \
  --config configs/gqa_acm.json \
  --gqa-root /path/to/gqa/processed \
  --vg-lmdb /path/to/lmdb/vg_obj36.lmdb \
  --testdev-lmdb /path/to/lmdb/testdev_obj36.lmdb \
  --clip-root /path/to/clip_features/gqa_rn50x4_grid7 \
  --official-checkpoint /path/to/model_LXRT.pth \
  --seed 9595 \
  --output outputs/gqa_acm_seed9595
```

## Evaluation & Visual Dependence Diagnostics

### VQA v2 (Joint, CLIP-only, and Fixed-Derangement Joint CLIP Swap)

```bash
python scripts/eval_vqa2.py \
  --checkpoint outputs/vqa2_acm_seed9595/epoch4.pt \
  --vqa-data-root /path/to/lxmert/data/vqa \
  --train-lmdb /path/to/lmdb/train2014_obj36.lmdb \
  --val-lmdb /path/to/lmdb/val2014_obj36.lmdb \
  --clip-root /path/to/clip_features/vqa2_rn50x4_grid7 \
  --official-checkpoint /path/to/model_LXRT.pth \
  --split minival \
  --run-clip-only \
  --run-clip-swap \
  --output outputs/eval_vqa2_acm_seed9595
```

### GQA (Joint, CLIP-only, CLIP Swap, Region Swap, and CLIP-only Swap)

```bash
python scripts/eval_gqa.py \
  --checkpoint outputs/gqa_acm_seed9595/epoch4.pth \
  --gqa-root /path/to/gqa/processed \
  --vg-lmdb /path/to/lmdb/vg_obj36.lmdb \
  --testdev-lmdb /path/to/lmdb/testdev_obj36.lmdb \
  --clip-root /path/to/clip_features/gqa_rn50x4_grid7 \
  --raw-questions /path/to/gqa/testdev_balanced_questions.json \
  --official-checkpoint /path/to/model_LXRT.pth \
  --output outputs/eval_gqa_acm_seed9595
```

## License & Acknowledgments

This repository is released under the MIT License (see `LICENSE`). The LXMERT backbone modules in `third_party/lxmert/` are adapted from [airsplay/lxmert](https://github.com/airsplay/lxmert), and CLIP grid features are extracted using [OpenAI CLIP](https://github.com/openai/CLIP).
