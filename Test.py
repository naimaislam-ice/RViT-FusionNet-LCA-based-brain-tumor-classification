import torch
import numpy as np

from Model import *
from Dataloader import *

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Re-instantiate the model and move to device
model = ResNet50ViT_LCDI(num_classes=4).to(device)

# 3. Load the best checkpoint (map to correct device)
checkpoint = torch.load('best_model.pt', map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# 4. Evaluate on test set
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs, _ = model(inputs)           # only classification logits
        preds = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")
