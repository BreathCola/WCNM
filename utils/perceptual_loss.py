"""VGG-16 feature-space perceptual loss used by Stage A."""

import torch
from torch import nn
import torch.nn.functional as F


class VGG16PerceptualLoss(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        features = vgg16(weights=weights).features.eval()
        boundaries = (4, 9, 16, 23)
        start = 0
        blocks = []
        for end in boundaries:
            blocks.append(features[start:end])
            start = end
        self.blocks = nn.ModuleList(blocks)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.ndim == 3:
            prediction = prediction.unsqueeze(0)
        if target.ndim == 3:
            target = target.unsqueeze(0)
        prediction = (prediction - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = prediction.new_zeros(())
        for block in self.blocks:
            prediction = block(prediction)
            with torch.no_grad():
                target = block(target)
            loss = loss + F.l1_loss(prediction, target)
        return loss / len(self.blocks)
