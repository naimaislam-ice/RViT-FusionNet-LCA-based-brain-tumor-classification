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

## Authors

- Naima Islam
- Sajeeb Kumar Ray
- Md. Anwar Hossain
- Syed Mohammed Shamsul Islam

---

# Key Contributions

This work contributes to the field of brain tumor classification in several ways:

- The proposed framework, **RViT-FusionNet**, combines the strengths of ResNet-50 and Vision Transformer (ViT) for brain tumor classification using MRI images through a **Local Cross-Attention (LCA)** module that fuses local texture representations with global contextual representations, improving spatial understanding and feature alignment.

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

## System Architecture of RViT-FusionNet

<p align="center">
  <img src="Visuals/System_Arcitecture_of_RViT-FusionNet.png" width="95%">
</p>



---

## Grad-CAM Visualization

<p align="center">
  <img src="Visuals/Grad-CAM_Localization_of_Brain_Tumor.png" width="90%">
</p>


