# ANRF-AISEHack-Phase-2-Theme-1-Flood-Detection-IBM
### Flood Inundation Mapping — 3-Class Semantic Segmentation

**ANRF AISEHack Phase 2 | Theme 1: Flood Detection (IBM)**  
West Bengal, India | Satellite Remote Sensing | Disaster Response

---

## Overview

This repository contains the solution developed for Phase 2 of the ANRF AISEHack competition, Theme 1: Flood Detection, in collaboration with IBM. The objective is pixel-level segmentation of satellite imagery into three classes:

- **Class 0** — No Flood
- **Class 1** — Flood (active inundation)
- **Class 2** — Water Body (pre-existing permanent or seasonal water)

The model ingests 6-channel multi-source satellite patches (SAR + multispectral optical) and produces per-pixel class predictions, submitted as run-length encoded (RLE) flood masks.

---

## Repository Structure

```
.
├── data/
│   ├── image/              # Training and test GeoTIFF patches (512x512, 6-band)
│   ├── label/              # Ground truth label TIFs (0/1/2 per pixel)
│   ├── split/              # train.txt, val.txt, test.txt, pred.txt
│   └── prediction/image/   # Test images without ground truth (for submission)
├── checkpoints/
│   └── best_flood_iou.keras   # Best model checkpoint (monitored on val Flood IoU)
├── viz/                    # Saved visualisation outputs
├── flood_segmentation_unetpp.ipynb   # Main training and inference notebook
├── submission.csv          # Final RLE submission file
└── README.md
```

---

## Data

**Source:** Satellite imagery over West Bengal, India (2024 monsoon season)  
**Format:** GeoTIFF patches, 512x512 pixels, 6 channels  
**Labels:** Single-band GeoTIFF with values {0, 1, 2}

| Channel | Name  | Sensor        |
|---------|-------|---------------|
| 0       | HH    | SAR (Sentinel-1) |
| 1       | HV    | SAR (Sentinel-1) |
| 2       | Green | Multispectral |
| 3       | Red   | Multispectral |
| 4       | NIR   | Multispectral |
| 5       | SWIR  | Multispectral |

Split files in `data/split/` list image stem IDs (without `_image.tif` suffix). The `pred.txt` split contains test images with no ground truth labels, used to generate the final submission.

---

## Methodology

### Architecture: UNet++ with Enhancements

A lightweight UNet++ encoder-decoder (~12M parameters) trained from scratch on the 6-channel input. Three enhancements are applied over the baseline:

**Dilated Convolution Blocks**  
Each encoder block applies two parallel convolutions — standard (dilation=1) and dilated (dilation=2) — whose outputs are summed before the SE gate. This enlarges the effective receptive field without downsampling.

**Squeeze-and-Excitation Channel Attention**  
After each conv block, a channel-wise attention gate (ratio=8) learns to re-weight feature maps by their global spatial response, enabling the network to emphasise flood-informative channels such as NIR and HV.

**ASPP Bottleneck**  
The encoder bottleneck uses Atrous Spatial Pyramid Pooling with dilation rates [1, 2, 4, 6] plus a global average pooling branch, capturing flood context at multiple spatial scales before decoding.

### Patch-Based Inference with Overlap Blending

Images are decomposed into 128x128 sub-patches with stride 96 (32-pixel overlap). During inference, softmax probability maps from overlapping patches are averaged in the stitching pass, eliminating boundary seam artifacts.

### Normalisation

A two-pass robust normalisation is applied per channel:
1. Clip to [1st percentile, 99th percentile] computed across the full training set
2. Z-score normalisation using Welford mean and standard deviation on the clipped values

SAR channels are not converted to dB. Raw linear amplitude with percentile clipping was found to outperform dB-converted inputs on this dataset.

### Loss Function

A four-component composite loss handles the severe class imbalance (no-flood pixels dominate approximately 73% of training labels):

| Component         | Weight | Role |
|-------------------|--------|------|
| Dice Loss         | 0.35   | Overlap-based, class-frequency invariant |
| Focal Loss (g=2)  | 0.25   | Down-weights easy background pixels |
| Weighted CE       | 0.15   | Per-pixel weights [0.074, 0.680, 0.247] |
| Tversky (a=0.7)   | 0.25   | Penalises missed flood pixels over false alarms |

### Augmentation

Geometric: random horizontal flip, vertical flip, 90-degree rotations (0/90/180/270).  
Photometric: contrast jitter on optical channels only (channels 2-5), multiplicative factor in [0.85, 1.15]. SAR channels are excluded from photometric augmentation.

---

## Training

| Hyperparameter     | Value                        |
|--------------------|------------------------------|
| Optimizer          | AdamW (weight decay = 1e-5)  |
| Learning rate      | 3e-4 with cosine decay       |
| LR minimum         | 1e-6                         |
| Batch size         | 16                           |
| Max epochs         | 80                           |
| Early stopping     | Patience 20, monitor val Flood IoU |
| Checkpoint metric  | val_IoU_Flood (maximise)     |
| Patch size         | 128x128                      |
| Stride             | 96                           |
| Flood threshold    | 0.45 (grid-searched on val)  |

Training is performed on a single NVIDIA GPU using TensorFlow 2 with mixed-precision disabled for stability.

---

## Results

Aggregate test-set metrics (images pid_070 to pid_079):

| Metric         | Score  |
|----------------|--------|
| IoU No Flood   | 0.7358 |
| IoU Flood      | 0.1338 |
| IoU Water Body | 0.2948 |
| mIoU           | 0.3882 |
| Pixel Accuracy | 69.3%  |

Best validation Flood IoU achieved: **0.1517** (epoch 8 checkpoint).  
Dynamic threshold search: best flood threshold = **0.45**.

---

## Submission Format

Predictions are submitted as a CSV with run-length encoded flood masks following Kaggle column-major convention (top-to-bottom, then left-to-right, 1-indexed). Images with no predicted flood pixels are encoded as `0 0`.

```
id,rle_mask
20240529_EO4_RES2_fl_pid_080,498 6 1008 7 ...
20240529_EO4_RES2_fl_pid_081,340 16 424 14 ...
```

---

## Requirements

```
tensorflow >= 2.12
rasterio
numpy
pandas
opencv-python
tqdm
matplotlib
```
Model Checkpoints: https://www.kaggle.com/models/adithyar21510/yolov8

Install dependencies:

```bash
pip install tensorflow rasterio numpy pandas opencv-python tqdm matplotlib
```

The notebook was developed and tested on `nvcr.io/nvidia/tensorflow:25.02-tf2-py3`.

---

## How to Run

1. Place training images in `data/image/`, labels in `data/label/`, and split files in `data/split/`.
2. Place test images (no ground truth) in `data/prediction/image/` and list their IDs in `data/split/pred.txt`.
3. Open and run `flood_segmentation_unetpp.ipynb` sequentially from top to bottom.
4. The final `submission.csv` will be written to the project root.

---

## References

- Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation," MICCAI 2018
- Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017
- Salehi et al., "Tversky Loss Function for Image Segmentation," MICCAI 2017
- IBM NASA Geospatial: https://huggingface.co/ibm-nasa-geospatial
- TerraTorch: https://github.com/terrastackai/terratorch
- AISEHack Edition 1 helper code: https://github.com/AISEHack/AISEHack_Edition1_2026

---

## License

This project is released under the ANRF Open License.  
License terms: https://anrfonline.in/ANRF/AbstractFilePath?FileType=E&FileName=OL_AISE.pdf&PathKey=DOCUMENT_TEMPLATE

---

## Acknowledgements

This work was developed as part of ANRF AISEHack Phase 2, Theme 1, supported by IBM and the National Research Foundation of India. The competition problem was designed around real flood events in West Bengal using data from Sentinel-1 and multispectral satellite platforms.
