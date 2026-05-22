# ============================================
# Visualisation_GradCAM.py
# ============================================

import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM

from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget
)

from pytorch_grad_cam.utils.image import (
    show_cam_on_image
)

from Model import *
from Test import *

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Model
model = ResNet50ViT_LCDI(num_classes=4).to(device)

checkpoint = torch.load(
    'best_model.pt',
    map_location=device
)

model.load_state_dict(checkpoint)

model.eval()

# GradCAM Wrapper
class GradCAMWrapper(torch.nn.Module):

    def __init__(self, model):

        super().__init__()

        self.model = model

    def forward(self, x):

        out, _ = self.model(x)

        return out

wrapped_model = GradCAMWrapper(model)

# Test predictions
all_preds = []
all_labels = []
all_images = []

with torch.no_grad():

    for images, labels in test_loader:

        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_images.extend(images.cpu())

class_names = [
    'glioma',
    'meningioma',
    'notumor',
    'pituitary'
]

mean = np.array([0.485, 0.456, 0.406])

std = np.array([0.229, 0.224, 0.225])

tumor_classes = [
    'glioma',
    'meningioma',
    'pituitary'
]

tumor_class_indices = [
    i for i, lbl in enumerate(all_labels)
    if class_names[lbl] in tumor_classes
]

if len(tumor_class_indices) < 20:

    sampled_indices = tumor_class_indices

else:

    sampled_indices = random.sample(
        tumor_class_indices,
        20
    )

def imshow_tensor(img_tensor):

    img = img_tensor.cpu().numpy().transpose(1, 2, 0)

    img = std * img + mean

    return np.clip(img, 0, 1)

target_layer = model.resnet_conv[-1]

cam = GradCAM(
    model=wrapped_model,
    target_layers=[target_layer]
)

fig, axes = plt.subplots(
    5,
    4,
    figsize=(28, 25),
    dpi=400
)

for i, idx in enumerate(sampled_indices):

    img_tensor = all_images[idx].to(device)

    img_np = imshow_tensor(img_tensor)

    true_label = all_labels[idx]

    outputs, _ = model(img_tensor.unsqueeze(0))

    probs = torch.nn.functional.softmax(
        outputs,
        dim=1
    ).detach().cpu().numpy()[0]

    pred_class = np.argmax(probs)

    targets = [ClassifierOutputTarget(pred_class)]

    grayscale_cam = cam(
        input_tensor=img_tensor.unsqueeze(0),
        targets=targets
    )[0]

    visualization = show_cam_on_image(
        img_np,
        grayscale_cam,
        use_rgb=True
    )

    ax = axes[i // 4, i % 4]

    ax.imshow(visualization)

    true_name = class_names[true_label]

    pred_name = class_names[pred_class]

    ax.set_title(
        f"Actual: {true_name} \nPredicted: {pred_name}",
        color='green',
        fontsize=15
    )

    tumor_probs = [
        (name, probs[class_names.index(name)])
        for name in tumor_classes
    ]

    subtitle = "\n".join([
        f"{name}: {prob*100:.1f}%"
        for name, prob in tumor_probs
    ])

    ax.set_xlabel(
        subtitle,
        color='blue',
        fontsize=12
    )

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_facecolor('white')

for j in range(len(sampled_indices), 20):

    ax = axes[j // 4, j % 4]

    ax.axis('off')

fig.patch.set_facecolor('white')

plt.suptitle(
    "Grad-CAM Tumor Localization",
    fontsize=24,
    color='green'
)

plt.tight_layout()

plt.subplots_adjust(top=0.92)

save_path = "gradcam_20samples_5x4.png"

fig.savefig(
    save_path,
    bbox_inches='tight',
    dpi=400,
    facecolor='white'
)

print(f"✅ Saved Grad-CAM grid to: {save_path}")

plt.show()