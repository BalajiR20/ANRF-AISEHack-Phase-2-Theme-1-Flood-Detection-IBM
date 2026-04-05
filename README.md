# Flood Segmentation via Patch-Based YOLOv8 Backbone + UNet++ Decoder

**Task:** Multi-class semantic segmentation of flood regions from 6-channel SAR/optical imagery  
**Classes:** `0` No-Flood · `1` Flood · `2` Water-Body  
**Competition:** ANRFAISEHack — Theme 1, Phase 2  
**Input:** 512×512 GeoTIFF images (6 channels: HH, HV, Green, Red, NIR, SWIR)

---

## Table of Contents

1. [Motivation & Evolution of Approach](#1-motivation--evolution-of-approach)
2. [Dataset Overview](#2-dataset-overview)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Patch-Based Strategy](#4-patch-based-strategy)
5. [Model Architecture](#5-model-architecture)
6. [6-Channel Input Adaptation](#6-6-channel-input-adaptation)
7. [Loss Functions](#7-loss-functions)
8. [Training Configuration](#8-training-configuration)
9. [Inference & Dynamic Thresholding](#9-inference--dynamic-thresholding)
10. [Submission Format](#10-submission-format)
11. [Reproducibility](#11-reproducibility)
12. [File Structure](#12-file-structure)

---

## 1. Motivation & Evolution of Approach

This work represents the third major iteration of the segmentation pipeline. The evolution from earlier approaches was driven by concrete failure modes observed at each stage.

### 1.1 Phase 0 — IBM Prithvi Foundation Model (Abandoned)

The initial approach leveraged the **IBM Prithvi** geospatial foundation model, a transformer-based architecture pre-trained on large-scale multi-spectral remote sensing imagery. While appealing in theory due to its domain-relevant pretraining, Prithvi was designed for fixed, large input resolutions. Applying it directly to 512×512 images caused substantial compute overhead and, more critically, produced **overly smooth segmentation boundaries**. The model's attention mechanisms, operating over coarse patch tokens, could not resolve fine-grained transitions between flooded and non-flooded regions at the pixel level.

### 1.2 Phase 1 — Custom U-Net (No Patching, Abandoned)

Following the Prithvi experiments, a custom U-Net architecture was trained end-to-end on the full 512×512 images. Although this improved boundary sharpness relative to Prithvi, two fundamental issues remained:

- **Resolution bottleneck:** Processing the full 512×512 input pushed the encoder to heavily downsample spatial features, causing the network to learn blurry, spatially imprecise representations. Flood pixels at object boundaries were frequently misclassified or smoothed over.
- **Data scarcity:** The dataset is small. Training a full encoder-decoder from scratch on full-resolution images led to rapid overfitting, with the model memorising training samples rather than learning generalisable flood patterns.

Both problems had the same root cause: **too large an effective receptive field relative to available training data**, resulting in segmentation masks that captured coarse flood extents but failed at pixel-precise boundaries.

### 1.3 Phase 2 — Patch-Based YOLOv8 Backbone + UNet++ (Current)

The current approach addresses both failure modes simultaneously through a **patch-and-stitch paradigm** combined with a pretrained, feature-rich encoder:

1. **Patching** reduces the effective input resolution per forward pass (512×512 → 256×256), forcing the model to reason locally and produce sharper, pixel-precise boundaries.
2. **Patching augments the dataset** — a single 512×512 image generates multiple overlapping 256×256 patches, dramatically increasing the number of unique training samples seen by the model.
3. The **YOLOv8-small CSPDarknet backbone**, pretrained on COCO, provides rich, transferable feature representations that prevent the overfitting observed with from-scratch U-Net training.
4. A **UNet++ decoder** with dense skip connections recovers spatial resolution and combines multi-scale features for precise boundary delineation.

---

## 2. Dataset Overview

| Split | Role |
|---|---|
| `train.txt` + `val.txt` | Combined training set (all labelled images) |
| `test.txt` | Held-out evaluation set (IoU measurement & threshold search) |
| `pred.txt` | Unlabelled images for competition submission |

**Image format:** 32-bit float GeoTIFF, 512×512 pixels, 6 spectral channels.  
**Label format:** Single-band GeoTIFF with integer class values {0, 1, 2}.

**Class distribution (estimated):** No-Flood is heavily dominant, driving the use of class-weighted and focal losses to avoid trivial solutions.

---

## 3. Pipeline Overview

```
512×512 GeoTIFF (6ch)
        │
        ▼
  Per-channel normalisation
  (mean/std + percentile clip)
        │
        ▼
  Patch extraction
  (256×256, stride=192, ~overlap=160px)
        │
        ▼
  Augmentation (train only)
  flip H/V, rot90, spectral jitter
        │
        ▼
  YOLOv8-small CSPDarknet Backbone
  (pretrained COCO, 3→6 ch adapted)
  → P3 (1/8), P4 (1/16), P5 (1/32)
        │
        ▼
  UNet++ Decoder
  (4-level, dense skip connections)
        │
        ▼
  Logits (256×256 × 3 classes)
        │
        ▼
  Softmax → Patch softmax stacks
        │
        ▼
  Average-blend stitch → 512×512
        │
        ▼
  Dynamic flood threshold search
        │
        ▼
  Final mask → RLE → submission.csv
```

---

## 4. Patch-Based Strategy

### 4.1 Why Patching?

Patching is the cornerstone design decision of this pipeline. The rationale is twofold:

**Boundary precision.** When a 512×512 image is processed as a whole, the encoder must compress spatial information across a large field of view at each downsampling stage. By the time features reach the decoder, fine boundary details between flooded and non-flooded pixels are largely lost. Processing 256×256 patches instead means the network sees a smaller region at full detail, producing crisper, pixel-wise accurate segmentation boundaries — directly addressing the smooth-boundary failure of both the Prithvi and vanilla U-Net approaches.

**Effective dataset augmentation.** With a small labelled dataset, generating many overlapping patches per image multiplies the number of distinct training samples. Given a stride of 192px on a 512×512 image, each image produces approximately **9 unique patches**, effectively scaling the dataset ~9× without introducing artificial label noise.

### 4.2 Patch Parameters

| Parameter | Value | Rationale |
|---|---|---|
| `FULL_SIZE` | 512 px | Native image resolution |
| `PATCH_SIZE` | 256 px | YOLOv8 default, local detail retention |
| `STRIDE` | 192 px | 64 px step → ~160 px overlap |

### 4.3 Patch Extraction

```python
def get_patch_coords(full_size=512, patch_size=256, stride=192):
    # Generates (row, col) top-left corners covering the full image,
    # including boundary-aligned extra patches to ensure full coverage.
```

Patches that would fall outside the image boundary are edge-aligned rather than padded, ensuring no artificial border artefacts are introduced into training.

### 4.4 Overlapping Stitch (Inference)

During inference, each patch produces a per-pixel softmax probability map. Overlapping patches are **average-blended** back onto the full 512×512 canvas:

```
acc[r:r+h, c:c+w]   += softmax_prediction
count[r:r+h, c:c+w] += 1
final_soft = acc / count
```

This averaging smooths prediction disagreements between overlapping patches and reduces boundary artefacts at patch seams.

---

## 5. Model Architecture

### 5.1 Encoder: YOLOv8-small CSPDarknet

The backbone is the first 10 layers (`model.model.model[:10]`) of a pretrained `yolov8s.pt` checkpoint, which implements a **Cross-Stage Partial (CSP) DarkNet**. Three feature pyramid levels are captured via forward hooks:

| Feature Map | Backbone Layer | Spatial Scale | Channels (YOLOv8-s) |
|---|---|---|---|
| P3 | Layer 4 | 1/8 of input | 128 |
| P4 | Layer 6 | 1/16 of input | 256 |
| P5 | Layer 9 | 1/32 of input | 512 |

For a 256×256 input, P3 resolves to 32×32, P4 to 16×16, P5 to 8×8. These multi-scale features are the primary input to the decoder.

**Differential learning rates** are applied during fine-tuning: the backbone is updated at a low rate (`BACKBONE_LR = 1e-5`) to preserve pretrained representations, while the decoder is updated at a higher rate (`LR = 2e-4`).

**Selective freezing** of the first `FREEZE_STAGES = 2` CSP stages prevents low-level texture features from drifting during early training epochs.

### 5.2 Decoder: UNet++

The decoder implements a **UNet++** (nested U-Net) architecture with dense skip connections. This differs from vanilla UNet by inserting intermediate convolutional nodes between encoder skip features and decoder upsampling paths, enabling the network to learn from feature maps at multiple semantic depths simultaneously.

```
Decoder channels: [256, 128, 64, 32]
Dropout: 0.1
Final head: Conv2d → NUM_CLASSES (3)
```

The 4-level decoder progressively upsamples P5 → P4 → P3 → full patch resolution, at each stage concatenating skip features from the corresponding encoder level and intermediate dense nodes.

### 5.3 Full Model Summary

```
Input: (B, 6, 256, 256)
  └─ YOLOBackbone (CSPDarknet, layers 0-9)
       ├─ P3: (B, 128, 32, 32)
       ├─ P4: (B, 256, 16, 16)
       └─ P5: (B, 512, 8, 8)
  └─ UNet++ Decoder (4-level, channels 256→128→64→32)
Output: (B, 3, 256, 256) logits
```

---

## 6. 6-Channel Input Adaptation

YOLOv8 was originally trained on 3-channel RGB images. The flood dataset uses 6-channel imagery (HH, HV, Green, Red, NIR, SWIR). The first convolutional layer is adapted at model initialisation:

```python
# Original: Conv2d(3 → out_ch)
# New:      Conv2d(6 → out_ch)

old_w = first_conv.weight.data          # (out, 3, kH, kW)
new_w = old_w.mean(dim=1, keepdim=True) \
             .repeat(1, 6, 1, 1)
new_w *= 3.0 / 6                        # preserve activation magnitude
```

This weight-averaging strategy initialises the new input kernels from the mean of the pretrained RGB kernels, preserving the expected activation scale and providing a better starting point than random initialisation.

**Channel normalisation** is computed per-channel from a 50% random sample of the training split:
- Z-score normalisation (mean/std)
- Percentile clipping at [1st, 99th] to suppress outliers common in SAR backscatter

---

## 7. Loss Functions

A composite loss function is used to handle class imbalance and encourage both region-level accuracy and precise boundary prediction:

```
L_total = 0.75 × Dice + 0.25 × Focal + 0.15 × Weighted CE
```

| Component | Weight | Purpose |
|---|---|---|
| Dice Loss | 0.75 | Region overlap, robust to class imbalance |
| Focal Loss (γ=2) | 0.25 | Down-weights easy negatives (dominant No-Flood class) |
| Weighted Cross-Entropy | 0.15 | Explicit class rebalancing |

**Class weights** are set to counter the No-Flood dominance:

| Class | Weight |
|---|---|
| 0 No-Flood | 0.0737 |
| 1 Flood | 0.6798 |
| 2 Water-Body | 0.2465 |

A Tversky loss (α=0.7, β=0.3) is also implemented and available as an optional component (`tv_w=0` by default) for configurations where false negatives on the Flood class are prioritised.

---

## 8. Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 20 |
| Batch size | 4 |
| Gradient accumulation steps | 4 (effective batch = 16) |
| Optimiser | AdamW |
| Decoder LR | 2e-4 |
| Backbone LR | 1e-5 |
| Minimum LR | 1e-6 |
| Weight decay | 1e-4 |
| LR schedule | Cosine Annealing |
| Mixed precision (AMP) | Enabled (FP16) |
| Gradient clipping | max_norm = 1.0 |

**Model selection:** The checkpoint with the highest validation **Flood IoU** (Class 1) is saved, reflecting the competition's primary evaluation metric.

**Augmentation** (training patches only):
- Horizontal flip (p=0.5)
- Vertical flip (p=0.5)
- 90° rotation (k ∈ {0,1,2,3}, p=0.75)
- Spectral jitter on optical channels (Green, Red, NIR, SWIR) ×U(0.85, 1.15) (p=0.5)

SAR channels (HH, HV) are excluded from spectral jitter as their backscatter values have physical meaning that is not photometric in nature.

---

## 9. Inference & Dynamic Thresholding

### 9.1 Patch-and-Stitch Inference

At inference time, a full 512×512 image is decomposed into 256×256 patches at the same stride as training (192 px). Each patch is passed through the model to produce class logits, which are resized back to 256×256 and converted to softmax probabilities. All patch predictions are accumulated and average-blended onto the full canvas.

### 9.2 Dynamic Flood Threshold

Rather than using a fixed argmax decision, a **dynamic threshold** is searched over the held-out test set to maximise Flood IoU (Class 1):

```python
thresholds = np.linspace(0.0, 1.0, 21)  # 21 candidate thresholds
best_thresh = argmax over thresholds of mean(Flood-IoU across test set)
```

The threshold is applied post-hoc to the softmax Flood probability map, overriding the argmax prediction for pixels where `P(Flood) ≥ threshold`.

### 9.3 Metrics

Intersection-over-Union (IoU) is computed per class:

```
IoU_c = TP_c / (TP_c + FP_c + FN_c)
mIoU  = mean(IoU_0, IoU_1, IoU_2)
```

The primary competition metric is **Flood IoU** (Class 1). mIoU is tracked as a secondary diagnostic.

---

## 10. Submission Format

Predictions are encoded as **Run-Length Encoding (RLE)** in column-major (Fortran) order, consistent with the competition specification:

```python
def mask_to_rle(mask_2d, target_class=1):
    binary = (mask_2d == target_class).astype(np.uint8)
    flat   = binary.flatten(order='F')   # column-major
    ...
    return ' '.join(f'{start} {length}' for ...)
```

The submission CSV contains two columns: `id` and `rle_mask`. A sanity-check round-trip decoder is included to verify RLE correctness before submission.

---

## 11. Reproducibility

All random seeds are fixed globally:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

Channel statistics (mean, std, 1st/99th percentiles) are computed once from a fixed 50% random sample of the training split and reused for all normalisation operations.

---

## 12. File Structure

```
notebook/
├── yolov8-patched.ipynb          # Full training + inference pipeline
└── README.md                     # This document

kaggle/working/
├── checkpoints/
│   └── best_yolo_flood_iou.pt    # Best model checkpoint (by Flood IoU)
├── viz/
│   └── training_curves.png       # Loss and IoU plots
└── submission.csv                # Final RLE submission

kaggle/input/.../data/
├── image/         # *_image.tif — labelled split
├── label/         # *_label.tif
├── prediction/image/  # *_image.tif — unlabelled prediction split
└── split/
    ├── train.txt
    ├── val.txt
    ├── test.txt
    └── pred.txt
```

---

## Dependencies

```
python >= 3.9
torch >= 2.0 (CUDA 11.8)
torchvision
ultralytics >= 8.2.0
rasterio
numpy
pandas
opencv-python
matplotlib
```

Install via:
```bash
pip install ultralytics rasterio opencv-python matplotlib pandas
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Citation / Acknowledgements

This pipeline was developed for the ANRFAISEHack — Theme 1 (Flood Segmentation) competition. The YOLOv8 backbone weights are sourced from [Ultralytics](https://github.com/ultralytics/ultralytics) and are pretrained on COCO. The UNet++ decoder design follows the original formulation by Zhou et al. (2018). IBM Prithvi geospatial foundation model experiments were conducted during Phase 0 of this work.
