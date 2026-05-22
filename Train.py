import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import json 

from tqdm import tqdm

from Dataloader import *
from Model import *

# 3. Training & Validation Loop with Model Checkpointing

model = ResNet50ViT_LCDI(num_classes=4).to('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = optim.AdamW([
    {'params': model.classifier.parameters(), 'lr':1e-3},
    {'params': model.fusion_gate.parameters(), 'lr':1e-3},
    {'params': model.local_cross_attn.parameters(), 'lr':5e-4},
    {'params': model.resnet_conv.parameters(), 'lr':1e-5},
    {'params': model.vit_backbone.parameters(), 'lr':1e-5},
], weight_decay=5e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

num_epochs = 50

cls_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
best_val_acc = 0.0  # track best validation accuracy

device = next(model.parameters()).device
for epoch in range(num_epochs):
    if epoch==10:
        for p in model.resnet_conv[-3:].parameters(): p.requires_grad=True; optimizer.param_groups[3]['lr']=1e-4
    if epoch==20:
        for p in model.vit_backbone.parameters(): p.requires_grad=True; optimizer.param_groups[4]['lr']=1e-4

    model.train()
    running_loss, correct, total = 0,0,0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        domain_lbl = torch.randint(0,2,(labels.size(0),1,1,1),dtype=torch.float).to(device)
        cls_out, dom_out = model(inputs)
        loss = cls_criterion(cls_out, labels)
        if dom_out is not None:
            loss += 0.2 * domain_criterion(dom_out, domain_lbl.expand_as(dom_out))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        running_loss += loss.item()*labels.size(0)
        preds = cls_out.argmax(dim=1)
        correct += (preds==labels).sum().item()
        total += labels.size(0)
    train_loss = running_loss/total; train_acc = correct/total

    model.eval()
    val_loss, val_correct, val_total = 0,0,0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            cls_out, _ = model(inputs)
            loss = cls_criterion(cls_out, labels)
            val_loss += loss.item()*labels.size(0)
            preds = cls_out.argmax(dim=1)
            val_correct += (preds==labels).sum().item()
            val_total += labels.size(0)
    val_loss /= val_total; val_acc = val_correct/val_total

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    scheduler.step()
    print(f"Epoch {epoch+1}/{num_epochs} ", 
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | ",
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
# Save training history as JSON
with open("history.json", "w") as f:
    json.dump(history, f, indent=4)