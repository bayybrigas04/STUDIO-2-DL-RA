"""Submission Studio 2 - Klasifikasi Sampah.

Arsitektur: ResNet-50 dengan head 5 kelas.

Kontrak (dipanggil oleh evaluate.py):
  - get_model(): hanya MEMBANGUN arsitektur (tanpa mengunduh bobot, tanpa memuat model.pth).
  - get_transform(): transform validasi, HARUS sama dengan yang dipakai saat training.
  - Output: logits [batch, 5], urutan kelas (alfabetis, sama dengan ImageFolder):
      0 botol_plastik, 1 gelas_plastik, 2 kertas_kardus, 3 saset_kemasan, 4 styrofoam
"""
import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 5
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]  # statistik ImageNet
STD = [0.229, 0.224, 0.225]


def get_model():
    model = models.resnet50(weights=None)  # weights=None: tidak mengunduh apa pun
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),  # ukuran akhir selalu 224x224
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])