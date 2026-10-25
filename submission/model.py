import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ConvNeXt_Tiny_Weights

ARCH = "convnext_tiny"
IMG_SIZE = 256
NUM_CLASSES = 5
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_model(pretrained=True, drop_path=0.1):
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.convnext_tiny(weights=weights, stochastic_depth_prob=drop_path)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])