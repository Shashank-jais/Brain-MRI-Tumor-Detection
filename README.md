
# Brain Tumor Classification and Segmentation with MONAI and PyTorch

## 🧠 Project Overview

This project focuses on **Brain Tumor Segmentation and Classification** from MRI scans using **deep learning** techniques.  
We leverage the **MONAI** (Medical Open Network for Artificial Intelligence) framework built on **PyTorch** to develop a high-quality, GPU-accelerated 3D medical image segmentation model.

The goal is to assist medical professionals by providing accurate tumor localization, classification, and segmentation for faster and more reliable diagnosis.

---

## 🎯 Objectives

- Build a **deep learning model** for brain tumor segmentation.
- **Classify** different tumor regions: Tumor Core (TC), Whole Tumor (WT), and Enhancing Tumor (ET).
- Utilize **3D multimodal MRI data** effectively.
- Achieve **high segmentation accuracy** with minimal manual preprocessing.
- Optimize **GPU-based training** using MONAI and PyTorch.

---

## 🗂️ Dataset Details

- **Source**: Medical Segmentation Decathlon – Task01_BrainTumour
- **Modality**: Multimodal MRI (T1, T1Gd, T2, FLAIR)
- **Labels**:
  - **Label 1**: Peritumoral edema
  - **Label 2**: GD-enhancing tumor
  - **Label 3**: Necrotic and non-enhancing tumor core
- **Derived Segmentation**:
  - **Tumor Core (TC)** = Label 2 + Label 3
  - **Whole Tumor (WT)** = Label 1 + Label 2 + Label 3
  - **Enhancing Tumor (ET)** = Label 2
- **Format**: NIfTI (.nii.gz)

---

## 🛠️ Environment Setup

### Install Required Libraries:
```bash
pip install monai-weekly[nibabel, tqdm]
pip install matplotlib
pip install onnxruntime
```

### Verify GPU Support for PyTorch:
```python
import torch
if torch.cuda.is_available():
    print(f"PyTorch is using the GPU: {torch.cuda.get_device_name(0)}")
```

> **Note**: A CUDA-capable GPU is highly recommended for efficient 3D training.

---

## 🔄 Data Preparation

- **Loading and Transformations** using MONAI pipelines.
- **Multi-Channel Labeling**:
  - Converts original labels into three output channels (TC, WT, ET).
- **Data Augmentation**:
  - Random cropping
  - Flipping
  - Intensity shifting
- **Normalization**:
  - Intensity normalization performed channel-wise.

---

## 🧩 Model Architecture: MONAI-UNet

### Key Components:

- **Encoder (Contracting Path)**: Extracts hierarchical features with downsampling.
- **Bottleneck**: Connects encoder and decoder, captures complex features.
- **Decoder (Expansive Path)**: Reconstructs image with upsampling + skip connections.
- **Skip Connections**: Preserve spatial details for finer segmentation boundaries.
- **Output Layer**: 1x1 convolution to map features to segmentation masks.

### Loss Functions:

- **Dice Loss** for segmentation accuracy.
- **Cross-Entropy Loss** for multi-class classification.

---

## ⚙️ Training Details

- **Batch Size**: 1
- **Caching**: Disabled (cache_rate = 0.0)
- **Workers**: 4
- **Data Splitting**: Random splits for training and validation sets.

---

## 📈 Results

| Metric | Value |
|:------:|:-----:|
| **Segmentation Accuracy** | **79.14%** |

- Model performs reliably across different tumor classes (TC, WT, ET).
- Generalizes well across the validation data.

---

## 📋 Project Structure

```
├── Brain_tumor_Code.ipynb        # Jupyter Notebook for training and evaluation
├── Brain_tumor_Project_report.pdf                # Detailed project report
├── README.md                 # Project documentation (this file)
└── environment.yml           # (Optional) Conda environment file for quick setup
```

---

## 🔥 Key Highlights

- Fully supports **3D medical image segmentation**.
- Optimized for **GPU acceleration**.
- Minimal manual preprocessing required.
- High adaptability for future medical segmentation tasks.

---

## 🙌 Contributors

- **Shashank Jaiswal** – [SAP ID: 500109929]

---

## 📜 License

This project is created for academic purposes under the course **Pattern Recognition and Anomaly Detection** at **University of Petroleum and Energy Studies (UPES)**.  
You can reuse or extend it with proper attribution.

---

## 🚀 Future Work (Optional Enhancements)

- Incorporate **attention mechanisms** to further improve segmentation.
- Train on **larger datasets** for better generalization.
- Deploy the model for **real-time clinical use**.
- Extend the pipeline to **tumor growth prediction**.

---
