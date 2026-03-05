# 🧠 NeuroSeg AI — Brain Tumor Segmentation

<div align="center">

![Brain Tumor Segmentation](https://img.shields.io/badge/AI-Brain%20Tumor%20Segmentation-blueviolet?style=for-the-badge&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?style=for-the-badge&logo=flask)

**An AI-powered web application for automated brain tumor detection and segmentation from multi-modal MRI scans using a custom U-Net deep learning model.**

[🚀 Features](#-features) • [🏗️ Architecture](#️-model-architecture) • [📦 Installation](#-installation) • [🖥️ Usage](#️-usage) • [📁 Project Structure](#-project-structure) • [📊 Results](#-results)

</div>

---

## 📖 Overview

**NeuroSeg AI** is a full-stack medical imaging web application that leverages deep learning to assist in the detection and segmentation of brain tumors from MRI scans. The system accepts all **four standard MRI modalities** (FLAIR, T1, T1ce, T2) and produces a precise tumor segmentation mask overlaid on the scan — all in under **3 seconds**.

This project bridges the gap between cutting-edge AI research and clinical usability by providing a clean, intuitive web interface that radiologists and researchers can use without any programming knowledge.

> ⚠️ **Disclaimer:** This tool is intended for **research and educational purposes only**. It is not a certified medical device and should not be used as a substitute for professional medical diagnosis.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **Multi-Modal Input** | Accepts FLAIR, T1, T1ce, and T2 MRI sequences simultaneously |
| 🔬 **NIfTI Support** | Handles both standard images (PNG/JPG) and medical NIfTI formats (`.nii`, `.nii.gz`) |
| ⚡ **Fast Inference** | Processes and segments tumors in under 3 seconds |
| 📊 **Quantitative Metrics** | Reports tumor burden percentage and processing time |
| 🎨 **Visual Overlay** | Generates color-coded tumor mask overlaid on FLAIR image |
| 🌐 **Web Interface** | Accessible through any modern browser — no installation of ML tools needed by the end user |
| 🖥️ **CPU & GPU Support** | Automatically uses CUDA GPU if available, falls back to CPU |

---

## 🏗️ Model Architecture

The core of NeuroSeg AI is a custom **2D U-Net** convolutional neural network, purpose-built for biomedical image segmentation.

```
Input: (Batch, 4 channels, 192×192)
         │
    ┌────▼────┐
    │ Encoder │  DoubleConv blocks with MaxPool downsampling
    │  (×3)   │  Channels: 24 → 48 → 96
    └────┬────┘
         │
    ┌────▼────┐
    │Bottleneck│  DoubleConv: 96 → 192 channels
    └────┬────┘
         │
    ┌────▼────┐
    │ Decoder │  ConvTranspose2d upsampling + Skip Connections
    │  (×3)   │  Channels: 192 → 96 → 48 → 24
    └────┬────┘
         │
    ┌────▼────┐
    │ Output  │  1×1 Conv → Binary Segmentation Mask
    └─────────┘
Output: (Batch, 1, 192×192)
```

### Key Design Choices
- **Skip connections** preserve spatial detail lost during downsampling
- **Batch Normalization** after every convolution for training stability
- **4-channel input** fuses all MRI modalities for richer feature extraction
- **Sigmoid activation** at output for binary tumor/no-tumor prediction
- **Threshold: 0.5** on probability map to generate final binary mask

---

## 📊 Results

| Metric | Value |
|---|---|
| 🎯 Dice Similarity Coefficient | **0.78 (78%)** |
| ⏱️ Inference Time | **< 3 seconds** |
| 📐 Input Resolution | **192 × 192 px** |
| 🗂️ Training Dataset | **BraTS 2020** |
| 🔢 Input Channels | **4 (FLAIR, T1, T1ce, T2)** |
| 📤 Output | **Binary segmentation mask** |

The model was trained on the **BraTS 2020 (Brain Tumor Segmentation)** benchmark dataset, which contains multi-institutional, multi-modal MRI scans with expert-annotated tumor regions.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Deep Learning** | PyTorch |
| **Web Framework** | Flask |
| **Image Processing** | Pillow (PIL), NumPy |
| **Medical Imaging** | NiBabel (NIfTI support) |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Fonts & Icons** | Google Fonts (Inter), Font Awesome |

---

## 📁 Project Structure

```
Brain Tumor Project/
│
└── App/
    ├── app.py                   # 🚀 Flask backend — routing, preprocessing, inference
    ├── unet.py                  # 🧠 U-Net model definition (DoubleConv + UNet2D)
    ├── requirements.txt         # 📦 Python dependencies
    ├── run_app.bat              # ▶️  Windows batch script to run the app
    ├── test_model.py            # 🧪 Script for standalone model testing
    ├── verify_prediction.py     # ✅ Script to verify prediction outputs
    │
    ├── Model/
    │   └── best_unet2d_brats 3rd(0.78).pth   # 💾 Trained model weights
    │
    ├── templates/
    │   ├── home.html            # 🏠 Landing page
    │   ├── analyze.html         # 🔬 Upload & analysis page
    │   ├── about.html           # ℹ️  About the project
    │   └── index.html           # 📋 Index page
    │
    ├── static/
    │   └── css/
    │       └── style.css        # 🎨 Global stylesheet
    │
    └── uploads/                 # 📂 Temporary folder for uploaded files
```

---

## 📦 Installation

### Prerequisites
- Python **3.8** or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA for faster inference

### Step 1: Clone the Repository

```bash
git clone https://github.com/malaikairshad54/Brain-Tumor-Segmentation.git
cd Brain-Tumor-Segmentation
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r App/requirements.txt
```

**Dependencies include:**
```
flask
torch
torchvision
numpy
nibabel
pillow
```

> 💡 **For GPU support**, install PyTorch with CUDA from the [official PyTorch website](https://pytorch.org/get-started/locally/) before running the above command.

---

## 🖥️ Usage

### Running the Web App

#### Option A — Using the Batch Script (Windows)
```bash
# From the App/ directory
run_app.bat
```

#### Option B — Using Python directly
```bash
cd App
python app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

### How to Use the App

```
Step 1  ──►  Go to the "Start Analysis" page
Step 2  ──►  Upload all 4 MRI modality files:
                 • FLAIR (.nii / .nii.gz / .png / .jpg)
                 • T1    (.nii / .nii.gz / .png / .jpg)
                 • T1ce  (.nii / .nii.gz / .png / .jpg)
                 • T2    (.nii / .nii.gz / .png / .jpg)
Step 3  ──►  Click "Analyze"
Step 4  ──►  View the segmented tumor overlay + metrics
```

### Supported File Formats

| Format | Extension | Notes |
|---|---|---|
| NIfTI | `.nii` | Standard medical imaging format |
| Compressed NIfTI | `.nii.gz` | Gzip-compressed NIfTI |
| PNG Image | `.png` | Grayscale brain slice image |
| JPEG Image | `.jpg` / `.jpeg` | Grayscale brain slice image |

---

## 🔧 How It Works

### Preprocessing Pipeline

```
1. File Upload  ──►  Detect format (NIfTI or standard image)
2. NIfTI        ──►  Load with nibabel → Extract middle 2D slice
   Standard     ──►  Open with PIL → Convert to grayscale
3. Center Crop  ──►  Crop/resize to 192×192 pixels
4. Stack        ──►  Combine 4 modalities → shape (4, 192, 192)
5. Tensor       ──►  Convert to PyTorch tensor → add batch dim (1, 4, 192, 192)
```

### Inference & Visualization

```
1. Model Forward Pass  ──►  U-Net generates logit map (1, 1, 192, 192)
2. Sigmoid             ──►  Convert logits to probability (0.0 → 1.0)
3. Threshold (0.5)     ──►  Binary mask: 1=tumor, 0=healthy
4. Overlay             ──►  Red semi-transparent mask on FLAIR grayscale
5. Metrics             ──►  Tumor pixel % + inference time calculated
6. Response            ──►  PNG overlay encoded as Base64 → rendered in browser
```

---

## 🌐 Web Pages

| Page | Route | Description |
|---|---|---|
| Home | `/` | Landing page with features overview and project stats |
| Analysis | `/analyze` | Main tool — upload MRI files and view results |
| About | `/about` | Detailed explanation of the technology and methodology |

---

## 📋 API Reference

### `POST /analyze`

Accepts multipart form data with 4 MRI files.

**Form Fields:**

| Field | Type | Description |
|---|---|---|
| `flair` | File | FLAIR MRI scan |
| `t1` | File | T1-weighted MRI scan |
| `t1ce` | File | T1 contrast-enhanced MRI scan |
| `t2` | File | T2-weighted MRI scan |

**Response:** HTML page with:
- `prediction_image` — Base64-encoded PNG of the tumor overlay
- `tumor_percentage` — Percentage of pixels classified as tumor
- `inference_time` — Total processing time in seconds
- `has_tumor` — Boolean flag indicating tumor presence

---

## 🧪 Testing

To verify the model loads and runs correctly:

```bash
cd App
python test_model.py
```

To verify the prediction output format:

```bash
cd App
python verify_prediction.py
```

---

## 🚧 Limitations

- The model is a **2D slice-based** segmenter; it does not use volumetric 3D context
- For 3D NIfTI volumes, only the **middle slice** is extracted and analyzed
- Best results are achieved with proper **BraTS-format** MRI data
- The model is trained on a **single tumor class** (whole tumor binary segmentation)
- Not validated for clinical use

---

## 🔮 Future Improvements

- [ ] 3D volumetric U-Net for full scan segmentation
- [ ] Multi-class tumor segmentation (core, enhancing, edema)
- [ ] DICOM file format support
- [ ] Batch processing for multiple patients
- [ ] Model explainability with Grad-CAM overlays
- [ ] Docker containerization for easy deployment

---

## 📚 References

- **U-Net Paper**: Ronneberger, O., Fischer, P., & Brox, T. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). MICCAI.
- **BraTS Dataset**: Menze et al. (2015). [The Multimodal Brain Tumor Image Segmentation Benchmark](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4833122/). IEEE TMI.
- **PyTorch**: [https://pytorch.org](https://pytorch.org)
- **NiBabel**: [https://nipy.org/nibabel](https://nipy.org/nibabel)

---

## 👩‍💻 Author

**Malaika Irshad**

[![GitHub](https://img.shields.io/badge/GitHub-malaikairshad54-181717?style=flat&logo=github)](https://github.com/malaikairshad54)

---

---

<div align="center">

Made with ❤️ for advancing AI in Medical Imaging

⭐ **Star this repo** if you found it useful!

</div>
