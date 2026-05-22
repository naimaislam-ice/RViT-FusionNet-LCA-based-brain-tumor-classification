# ============================================
# Visualisation_History.py
# ============================================

import json
import matplotlib.pyplot as plt

# Load history from JSON
with open("history.json", "r") as f:
    history = json.load(f)

# Style similar to provided figure
plt.style.use('default')

# Create figure
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

epochs = range(1, len(history['train_loss']) + 1)

# ============================================
# Loss Plot
# ============================================

axs[0].plot(
    epochs,
    history['train_loss'],
    color='blue',
    linewidth=2,
    label='Train Loss'
)

axs[0].plot(
    epochs,
    history['val_loss'],
    color='blue',
    linestyle='--',
    linewidth=2,
    label='Validation Loss'
)

axs[0].set_title(
    'Training vs Validation Loss',
    fontsize=18,
    fontweight='bold'
)

axs[0].set_xlabel('Epoch', fontsize=14)

axs[0].set_ylabel('Loss', fontsize=14)

axs[0].legend(fontsize=12)

axs[0].grid(
    True,
    linestyle='--',
    alpha=0.5
)

# ============================================
# Accuracy Plot
# ============================================

axs[1].plot(
    epochs,
    history['train_acc'],
    color='green',
    linewidth=2,
    label='Train Accuracy'
)

axs[1].plot(
    epochs,
    history['val_acc'],
    color='green',
    linestyle='--',
    linewidth=2,
    label='Validation Accuracy'
)

axs[1].set_title(
    'Training vs Validation Accuracy',
    fontsize=18,
    fontweight='bold'
)

axs[1].set_xlabel('Epoch', fontsize=14)

axs[1].set_ylabel('Accuracy', fontsize=14)

axs[1].legend(fontsize=12)

axs[1].grid(
    True,
    linestyle='--',
    alpha=0.5
)

# ============================================
# Layout & Save
# ============================================

plt.tight_layout()

plt.savefig(
    "Training_and_Validation_Metrics.png",
    dpi=400,
    bbox_inches='tight'
)

print("✅ Saved: Training_and_Validation_Metrics.png")

plt.show()