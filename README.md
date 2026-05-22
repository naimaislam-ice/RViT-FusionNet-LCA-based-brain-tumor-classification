# RViT-FusionNet
## A Local Cross-Attention Feature Fusion-based Hybrid Framework for Brain Tumor Classification

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Torchvision](https://img.shields.io/badge/Torchvision-0.15+-orange)
![Status](https://img.shields.io/badge/Status-Research-green)

---

## Overview

RViT-FusionNet is a hybrid deep learning framework for brain tumor classification using MRI images. The framework integrates convolutional neural networks and transformer-based global representation learning through a Local Cross-Attention fusion strategy.

The proposed architecture combines:
- ResNet50 local spatial representation learning
- Vision Transformer global contextual learning
- Local Cross-Attention feature fusion
- Patch-based domain discrimination
- Adaptive gated multimodal fusion

The framework is designed to improve classification robustness, feature generalization, and explainability for MRI-based brain tumor diagnosis.

---

## Authors

- Naima Islam
- Sajeeb Kumar Ray
- Md. Anwar Hossain
- Syed Mohammed Shamsul Islam

---

# Key Contributions

This work contributes to the field of brain tumor classification in several ways:

- The proposed framework, **RViT-FusionNet**, combines the strengths of ResNet-50 and Vision Transformer for brain tumor classification using MRI images through a **Local Cross-Attention (LCA)** module that fuses local texture representations with global contextual representations, improving spatial understanding and feature alignment.

- A **patch-based domain discriminator** is introduced using adversarial learning with binary cross-entropy loss to derive domain-invariant spatial features from ResNet-derived feature maps, improving robustness and generalization across different MRI scanners.

- Multiple publicly available brain MRI datasets are utilized to comprehensively evaluate the proposed framework while maintaining strict separation of testing data during training.

- The framework incorporates **Grad-CAM visualization** to localize tumor regions influencing model predictions, improving transparency and trustworthiness in clinical decision support.

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

# Proposed Architecture

## System Architecture

<p align="center">
  <img src="assets/system_architecture.png" width="95%">
</p>

> Replace `assets/system_architecture.png` with your uploaded system architecture image path.

---

## Grad-CAM Visualization

<p align="center">
  <img src="assets/Grad-CAM Localization of Brain Tumor.png" width="90%">
</p>

> Replace `assets/gradcam_visualization.png` with your uploaded Grad-CAM visualization image path.
