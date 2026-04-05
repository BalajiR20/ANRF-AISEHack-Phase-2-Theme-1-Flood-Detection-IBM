# Flood Inundation Mapping using Deep Learning

## 3-Class Semantic Segmentation of Satellite Imagery

**ANRF AISEHack 2026 — Phase 2**
**Theme 1: Flood Detection (IBM)**
**Domain:** Remote Sensing, Computer Vision, Disaster Response

---

## 1. Overview

Flood disasters significantly impact infrastructure, agriculture, and human life. Rapid and accurate flood mapping is essential for emergency response, resource allocation, and long-term risk mitigation.

This project presents a deep learning–based system for pixel-level semantic segmentation of multi-source satellite imagery to detect flooded regions. The model processes multi-channel satellite data and produces high-resolution flood maps suitable for disaster response workflows.

The solution was developed as part of the **ANRF AISEHack Phase 2 Flood Detection Challenge**, organized in collaboration with IBM and the National Research Foundation of India.

---

## 2. Problem Definition

The objective of this project is to perform semantic segmentation of satellite imagery into three distinct classes:

| Class ID | Description                        |
| -------- | ---------------------------------- |
| 0        | No Flood                           |
| 1        | Flood (Active Inundation)          |
| 2        | Water Body (Permanent or Seasonal) |

The model operates at pixel level, enabling precise identification of flooded areas and supporting operational disaster management systems.

---

## 3. Dataset Description

**Geographic Region:** West Bengal, India
**Time Period:** 2024 Monsoon Season
**Data Type:** Multi-source satellite imagery
**Task:** Semantic segmentation

### Input Data Characteristics

| Property           | Value               |
| ------------------ | ------------------- |
| Image format       | GeoTIFF             |
| Image resolution   | 512 × 512 pixels    |
| Number of channels | 6                   |
| Label format       | Single-band GeoTIFF |
| Label values       | {0, 1, 2}           |

### Satellite Channels

| Index | Channel | Sensor Type                               |
| ----- | ------- | ----------------------------------------- |
| 0     | HH      | Sentinel-1 Synthetic Aperture Radar (SAR) |
| 1     | HV      | Sentinel-1 Synthetic Aperture Radar (SAR) |
| 2     | Green   | Multispectral                             |
| 3     | Red     | Multispectral                             |
| 4     | NIR     | Multispectral                             |
| 5     | SWIR    | Multispectral                             |

---

## 4. Methodology

### 4.1 Model Architecture

The system uses a lightweight **UNet++ encoder–decoder architecture** designed for semantic segmentation of satellite imagery. The architecture was modified to improve performance on flood detection tasks involving heterogeneous sensor data.

Key architectural components include:

#### Dilated Convolution Blocks

Each encoder block contains parallel standard and dilated convolutions. This increases the effective receptive field without reducing spatial resolution, allowing the model to capture contextual flood patterns.

#### Channel Attention using Squeeze-and-Excitation

A channel-wise attention mechanism dynamically re-weights feature maps based on global spatial information. This enables the model to prioritize flood-relevant spectral channels such as NIR and SAR.

#### Atrous Spatial Pyramid Pooling (ASPP)

The bottleneck layer incorporates multi-scale atrous convolutions to capture spatial context at different resolutions. This improves segmentation accuracy for both large and small flood regions.

---

### 4.2 Patch-Based Inference

Large satellite images are processed using overlapping patches to improve prediction consistency and reduce boundary artifacts.

| Parameter  | Value     |
| ---------- | --------- |
| Patch size | 128 × 128 |
| Stride     | 96        |
| Overlap    | 32 pixels |

Softmax probability maps from overlapping patches are averaged during reconstruction.

---

### 4.3 Data Normalization

A robust two-step normalization strategy was applied independently to each channel:

1. Percentile clipping to the range: 1st percentile to 99th percentile
2. Z-score normalization using dataset-wide mean and standard deviation

This approach improves numerical stability and reduces sensitivity to sensor noise.

---

### 4.4 Loss Function Design

Flood detection datasets typically exhibit strong class imbalance, with non-flood pixels dominating the image. To address this, a composite loss function was implemented.

| Loss Component         | Purpose                                |
| ---------------------- | -------------------------------------- |
| Dice Loss              | Improves segmentation overlap          |
| Focal Loss             | Focuses training on difficult examples |
| Weighted Cross Entropy | Handles class imbalance                |
| Tversky Loss           | Penalizes missed flood detections      |

---

## 5. Data Augmentation

To improve generalization, the following augmentation techniques were applied during training.

### Geometric Transformations

* Horizontal flip
* Vertical flip
* 90-degree rotation

### Photometric Transformations

Applied only to optical channels:

* Contrast variation
* Intensity scaling

Synthetic Aperture Radar channels were excluded from photometric augmentation to preserve signal characteristics.

---

## 6. Training Configuration

| Parameter               | Value                |
| ----------------------- | -------------------- |
| Optimizer               | AdamW                |
| Learning rate           | 3 × 10⁻⁴             |
| Learning rate scheduler | Cosine decay         |
| Minimum learning rate   | 1 × 10⁻⁶             |
| Batch size              | 16                   |
| Maximum epochs          | 80                   |
| Early stopping          | Enabled              |
| Evaluation metric       | Validation Flood IoU |
| Hardware                | NVIDIA GPU           |
| Framework               | TensorFlow           |

---

## 7. Model Performance

### Test Set Metrics

| Metric           | Score  |
| ---------------- | ------ |
| IoU (No Flood)   | 0.7358 |
| IoU (Flood)      | 0.1338 |
| IoU (Water Body) | 0.2948 |
| Mean IoU         | 0.3882 |
| Pixel Accuracy   | 69.3%  |

### Best Validation Performance

* Best Flood IoU: **0.1517**
* Training epoch: **8**
* Optimal threshold: **0.45**

---

## 8. System Workflow

```
Satellite Image
        ↓
Preprocessing and Normalization
        ↓
Patch Generation
        ↓
UNet++ Segmentation Model
        ↓
Overlap Blending
        ↓
Flood Mask Prediction
        ↓
Run-Length Encoding
        ↓
Submission File
```

---

## 9. Project Structure

```
project/
│
├── data/
│   ├── image/
│   ├── label/
│   ├── prediction/
│   └── split/
│
├── models/
│
├── notebooks/
│   └── flood_segmentation_unetpp.ipynb
│
├── utils/
│
├── submission.csv
│
└── README.md
```

---

## 10. Requirements

Install dependencies:

```bash
pip install tensorflow rasterio numpy pandas opencv-python tqdm matplotlib
```

---

## 11. Execution Instructions

### Step 1 — Prepare Dataset

```
data/image/
data/label/
data/split/
```

### Step 2 — Add Test Images

```
data/prediction/image/
data/split/pred.txt
```

### Step 3 — Run Training or Inference

```
flood_segmentation_unetpp.ipynb
```

### Step 4 — Generate Submission File

The output file will be:

```
submission.csv
```

---

## 12. Applications

This system can be used in:

* Disaster response and emergency management
* Flood risk monitoring
* Climate and environmental analysis
* Infrastructure protection
* Satellite-based early warning systems

---

## 13. References

* Zhou et al. — UNet++: A Nested U-Net Architecture for Medical Image Segmentation
* Lin et al. — Focal Loss for Dense Object Detection
* Salehi et al. — Tversky Loss Function for Image Segmentation
* IBM NASA Geospatial
* TerraTorch

---

## 14. License

This project is released under the ANRF Open License.
