import torch
import torch.nn as nn

from torchvision import models
from torchvision.ops import StochasticDepth


# --------------------------
# 2. Model Definition
# --------------------------

class LocalCrossAttention(nn.Module):

    def __init__(self, resnet_feat_dim=2048, vit_dim=768, num_heads=8):
        super().__init__()

        self.resnet_proj = nn.Conv2d(resnet_feat_dim, vit_dim, 1)

        self.cross_attn = nn.MultiheadAttention(
            vit_dim,
            num_heads,
            batch_first=True
        )

    def forward(self, res_conv, vit_patches):

        res_proj = self.resnet_proj(res_conv)

        B, C, H, W = res_proj.shape

        res_patches = res_proj.view(B, C, -1).permute(0, 2, 1)

        attn_out, _ = self.cross_attn(
            res_patches,
            vit_patches,
            vit_patches,
            need_weights=False
        )

        return attn_out.mean(dim=1)


class PatchDiscriminator(nn.Module):

    def __init__(self, in_channels=2048):
        super().__init__()

        self.conv_layers = nn.Sequential(

            nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 1, 1)
        )

    def forward(self, res_conv):

        return self.conv_layers(res_conv)


class ResNet50ViT_LCDI(nn.Module):

    def __init__(self, num_classes=4, fusion_dim=1024):

        super().__init__()

        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )

        self.resnet_conv = nn.Sequential(
            *list(resnet.children())[:-2]
        )

        self.resnet_pool = nn.AdaptiveAvgPool2d((1, 1))

        vit = models.vit_b_16(
            weights=models.ViT_B_16_Weights.IMAGENET1K_V1
        )

        vit.heads = nn.Identity()

        for block in vit.encoder.layers:
            block.drop_path = StochasticDepth(
                p=0.1,
                mode="batch"
            )

        self.vit_backbone = vit

        self.local_cross_attn = LocalCrossAttention()

        self.domain_discriminator = PatchDiscriminator()

        self.res_fuse = nn.Linear(2048, fusion_dim)

        self.vit_fuse = nn.Linear(vit.hidden_dim, fusion_dim)

        self.local_fuse = nn.Linear(768, fusion_dim)

        self.fusion_gate = nn.Sequential(

            nn.Linear(fusion_dim * 3, 128),
            nn.ReLU(),

            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(

            nn.Dropout(0.5),

            nn.Linear(fusion_dim, 512),

            nn.GELU(),

            nn.Dropout(0.3),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):

        res_conv = self.resnet_conv(x)

        res_global = self.resnet_pool(res_conv).view(
            x.size(0),
            -1
        )

        vit_patches = self.vit_backbone._process_input(x)

        local_feat = self.local_cross_attn(
            res_conv,
            vit_patches
        )

        domain_logits = (
            self.domain_discriminator(res_conv)
            if self.training else None
        )

        res_p = self.res_fuse(res_global)

        vit_p = self.vit_fuse(
            vit_patches.mean(dim=1)
        )

        local_p = self.local_fuse(local_feat)

        gate = self.fusion_gate(
            torch.cat([res_p, vit_p, local_p], dim=1)
        )

        fused = (
            gate[:, 0:1] * res_p +
            gate[:, 1:2] * vit_p +
            gate[:, 2:3] * local_p
        )

        out = self.classifier(fused)

        return out, domain_logits


# =========================
# Model Initialization
# =========================

num_classes = 4

model = ResNet50ViT_LCDI(num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# =========================
# Parameter Count
# =========================

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")