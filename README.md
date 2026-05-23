# RViT-FusionNet
## A Local Cross-Attention Feature Fusion-based Hybrid Framework for Brain Tumor Classification

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Torchvision](https://img.shields.io/badge/Torchvision-0.15+-orange)
![Status](https://img.shields.io/badge/Status-Research-green)

---

## Overview

<p align="center">
  <img src="Visuals/Overview_of_the_ proposed_RViT-FusionNet_method.png" width="95%">
</p>

RViT-FusionNet is a hybrid deep learning framework for brain tumor classification using MRI images. The framework integrates convolutional neural networks and transformer-based global representation learning through a Local Cross-Attention fusion strategy.

The proposed architecture combines:

- ResNet50-based local spatial feature extraction
- Vision Transformer (ViT)-based global contextual representation learning
- Local Cross-Attention (LCA) feature fusion
- Adversarial domain-invariant feature learning through a domain discriminator
- Adaptive gated feature fusion
- Grad-CAM-based tumor localization

The framework is designed to improve classification robustness, feature generalization, and explainability for MRI-based brain tumor diagnosis.

---

# Proposed Architecture

## System Architecture of RViT-FusionNet

<p align="center">
  <img src="Visuals/System_Arcitecture_of_RViT-FusionNet.png" width="95%">
</p>

---

# Key Contributions

This work contributes to the field of brain tumor classification in several ways:

- The proposed framework, **RViT-FusionNet**, combines the strengths of ResNet-50 and Vision Transformer (ViT) for brain tumor classification using MRI images through a **Local Cross-Attention (LCA)** module that fuses local texture representations with global contextual representations, improving spatial understanding and feature alignment.

- A **patch-based domain discriminator** is introduced using adversarial learning with binary cross-entropy loss to derive domain-invariant spatial features from ResNet-derived feature maps, improving robustness and generalization across different MRI scanners.

- Multiple publicly available brain MRI datasets are utilized to comprehensively evaluate the proposed framework while maintaining strict separation of testing data during training.

- The framework incorporates **Grad-CAM visualization** to localize tumor regions influencing model predictions, improving transparency and trustworthiness in clinical decision support.

---

## Dataset

The proposed **RViT-FusionNet** framework was evaluated using four publicly available brain MRI datasets (DS1–DS4) containing four tumor categories:

- Glioma
- Meningioma
- Pituitary
- No Tumor

### Dataset Summary

| Dataset | Description | Total Images | Usage |
|----------|-------------|-------------:|--------|
| DS1 | Combined Figshare, SARTAJ, and Br35H datasets | 7,023 | Training + Testing |
| DS2 | Extended version of DS1 with additional images | 7,153 | External Evaluation |
| DS3 | Brain tumor MRI dataset | 3,264 | External Evaluation |
| DS4 | Brain Tumor MRI Classification dataset | 40,100 | External Evaluation |

### Training Strategy

- RViT-FusionNet was trained using the **training subset of DS1**.
- The best model was selected according to validation performance during training.
- The selected model was saved and evaluated on:
  - DS1 test set
  - DS2
  - DS3
  - DS4
- External datasets were used to evaluate the model's **generalization capability across different MRI distributions**.

### Dataset Sources

- DS1: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

- DS2: [Brain Tumor MRI Data](https://www.kaggle.com/datasets/tombackert/brain-tumor-mri-data)

- DS3: [Brain Tumor Classification MRI](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/data)

- DS4: [Brain Tumor MRI Classification](https://doi.org/10.34740/kaggle/dsv/10329066)


---

## Experimental Results

| Dataset | Class | Precision | Recall | F1-Score | Accuracy |
|----------|--------|------------|---------|-----------|-----------|
| DS1 | Glioma | 1.00 | 0.99 | 0.99 | 99.08% |
|  | Meningioma | 0.98 | 0.98 | 0.98 |  |
|  | No Tumor | 0.99 | 1.00 | 1.00 |  |
|  | Pituitary | 0.99 | 1.00 | 1.00 |  |
| DS2 | Glioma | 1.00 | 0.99 | 1.00 | 99.66% |
|  | Meningioma | 0.99 | 0.99 | 0.99 |  |
|  | No Tumor | 1.00 | 1.00 | 1.00 |  |
|  | Pituitary | 1.00 | 1.00 | 1.00 |  |
| DS3 | Glioma | 1.00 | 0.90 | 0.95 | 96.20% |
|  | Meningioma | 0.93 | 0.99 | 0.96 |  |
|  | No Tumor | 0.92 | 1.00 | 0.95 |  |
|  | Pituitary | 1.00 | 0.98 | 0.99 |  |
| DS4 | Glioma | 0.98 | 0.92 | 0.94 | 95.05% |
|  | Meningioma | 0.91 | 0.95 | 0.93 |  |
|  | No Tumor | 0.96 | 0.99 | 0.97 |  |
|  | Pituitary | 0.95 | 0.93 | 0.94 |  |


---

## RViT-FusionNet model’s Accurate Predictions 

<p align="center">
  <img src="Visuals/Correct_Predictions.png" width="90%">
</p>

---

## Grad-CAM Visualization

<p align="center">
  <img src="Visuals/Grad-CAM_Localization_of_Brain_Tumor.png" width="90%">
</p>

---

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{Islam2026RViTFusionNet,
  title={RViT-FusionNet: A Local Cross-Attention Feature Fusion-based Hybrid Framework for Brain Tumor Classification},
  author={Naima Islam and Sajeeb Kumar Ray and Md. Anwar Hossain and Syed Mohammed Shamsul Islam},
  journal={Under Review},
  year={2026}
}
```

## Authors

- Naima Islam
- Sajeeb Kumar Ray
- Md. Anwar Hossain
- Syed Mohammed Shamsul Islam

---

# Repository Contents

This repository includes:

- Original Jupyter Notebook implementation
- Modular Python implementation
- Trained best model weights
- Training history
- Visualization outputs
- Grad-CAM outputs
- Confusion matrix outputs
- Prediction visualization outputs

---
