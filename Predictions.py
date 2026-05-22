# ============================================
# Visualisation_Predictions.py
# ============================================

import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict

from Model import *
from Test import *

# ============================================
# Device
# ============================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================
# Load Best Model
# ============================================

model = ResNet50ViT_LCDI(num_classes=4).to(device)

checkpoint = torch.load(
    'best_model.pt',
    map_location=device
)

model.load_state_dict(checkpoint)

model.eval()

# ============================================
# Collect Predictions
# ============================================

all_preds = []

all_labels = []

all_images = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())

        all_labels.extend(labels.cpu().numpy())

        all_images.extend(images.cpu())

# ============================================
# Class Names
# ============================================

class_names = [
    'glioma',
    'meningioma',
    'notumor',
    'pituitary'
]

# ============================================
# Helper Functions
# ============================================

def imshow_tensor(img, mean, std):

    img = img.numpy().transpose((1, 2, 0))

    img = std * img + mean

    img = np.clip(img, 0, 1)

    return img

# ============================================
# Get Prediction Probabilities
# ============================================

def get_probs(image_tensor):

    input_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(input_tensor)

        if isinstance(output, tuple):
            output = output[0]

        probs = F.softmax(
            output,
            dim=1
        ).squeeze().cpu().numpy()

    return probs

# ============================================
# Visualization Function
# ============================================

def show_image_predictions(
    indices,
    title,
    color,
    save_path=None
):

    # ========================================
    # 3 × 3 Grid
    # ========================================

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(14, 14),
        dpi=400
    )

    mean = np.array([
        0.485,
        0.456,
        0.406
    ])

    std = np.array([
        0.229,
        0.224,
        0.225
    ])

    axes = axes.flatten()

    for i, idx in enumerate(indices[:9]):

        image = all_images[idx]

        true_label = class_names[all_labels[idx]]

        probs = get_probs(image)

        pred_label = class_names[np.argmax(probs)]

        ax = axes[i]

        ax.imshow(
            imshow_tensor(
                image,
                mean,
                std
            )
        )

        # ====================================
        # Prediction Title
        # ====================================

        ax.set_title(
            f"Actual: {true_label}\nPredicted: {pred_label}",
            color=color,
            fontsize=9,
            fontweight='bold',
            pad=4
        )

        # ====================================
        # Confidence Scores
        # ====================================

        subtitle = "\n".join([

            f"{class_names[j]}: {probs[j]*100:.1f}%"

            for j in range(len(class_names))
        ])

        ax.text(
            0.5,
            -0.08,
            subtitle,
            transform=ax.transAxes,
            fontsize=6,
            color='blue',
            ha='center',
            va='top'
        )

        ax.set_xticks([])

        ax.set_yticks([])

    # ========================================
    # Hide Empty Subplots
    # ========================================

    for j in range(len(indices[:9]), 9):

        axes[j].axis('off')

    # ========================================
    # Figure Title
    # ========================================

    plt.suptitle(
        title,
        fontsize=18,
        color=color,
        fontweight='bold'
    )

    # ========================================
    # Tight Compact Layout
    # ========================================

    plt.subplots_adjust(
        hspace=0.45,
        wspace=0.15,
        top=0.92,
        bottom=0.04
    )

    # ========================================
    # Save Figure
    # ========================================

    if save_path:

        fig.savefig(
            save_path,
            bbox_inches='tight',
            dpi=400
        )

        print(f"✅ Saved figure to: {save_path}")

    plt.show()

# ============================================
# Stratified Sampling
# ============================================

def stratified_sample(
    indices,
    labels,
    samples_per_class=3
):

    class_to_indices = defaultdict(list)

    for idx in indices:

        class_to_indices[labels[idx]].append(idx)

    sampled_indices = []

    for class_label in class_to_indices:

        sampled = random.sample(
            class_to_indices[class_label],
            min(
                samples_per_class,
                len(class_to_indices[class_label])
            )
        )

        sampled_indices.extend(sampled)

    return sampled_indices

# ============================================
# Correct / Wrong Indices
# ============================================

correct_indices = [

    i for i, (p, l)

    in enumerate(zip(all_preds, all_labels))

    if p == l
]

wrong_indices = [

    i for i, (p, l)

    in enumerate(zip(all_preds, all_labels))

    if p != l
]

# ============================================
# Correct Predictions
# ============================================

# ============================================
# Correct Predictions (3 × 3)
# ============================================

if correct_indices:

    balanced_correct_indices = stratified_sample(
        correct_indices,
        all_labels,
        samples_per_class=3
    )

    # Ensure maximum 9 samples
    balanced_correct_indices = balanced_correct_indices[:9]

    show_image_predictions(
        balanced_correct_indices,
        "Correct Predictions",
        color='green',
        save_path="balanced_correct_predictions.png"
    )

# ============================================
# Wrong Predictions
# ============================================

if wrong_indices:

    show_image_predictions(
        wrong_indices[:9],
        "Wrong Predictions",
        color='red',
        save_path="wrong_predictions.png"
    )