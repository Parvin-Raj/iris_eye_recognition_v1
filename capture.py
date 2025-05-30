import torch
import torch.nn as nn
from torchvision import models

class IrisEncoder(nn.Module):
    def __init__(self, num_classes):
        super(IrisEncoder, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
