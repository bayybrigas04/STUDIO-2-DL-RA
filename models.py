"""Kandidat backbone untuk eksperimen (BUKAN bagian submission).

Dipakai saat training: pretrained=True memuat bobot ImageNet.
Kalau sebuah kandidat menang, pindahkan definisinya ke submission/model.py
dengan pretrained=False (get_model tidak boleh mengunduh bobot).
"""
import timm
from torchvision import models
import torch.nn as nn

NUM_CLASSES = 5

CANDIDATES = {
    "resnet50":        "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "convnext_tiny":   "convnext_tiny",
}


def build(name, pretrained=True):
    return timm.create_model(CANDIDATES[name], pretrained=pretrained,
                             num_classes=NUM_CLASSES)