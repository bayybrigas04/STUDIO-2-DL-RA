"""Submission Studio 2 - Klasifikasi Sampah.

Arsitektur: ConvNeXt-Tiny (torchvision) dengan head 5 kelas.

Kontrak (dipanggil oleh evaluate.py):
  - get_model(): hanya MEMBANGUN arsitektur (tanpa mengunduh bobot, tanpa memuat model.pth).
  - get_transform(): transform validasi, HARUS sama dengan yang dipakai saat training.
  - Output: logits [batch, 5], urutan kelas (alfabetis, sama dengan ImageFolder):
      0 botol_plastik, 1 gelas_plastik, 2 kertas_kardus, 3 saset_kemasan, 4 styrofoam

Catatan: nama dan bentuk parameter harus identik dengan yang dipakai saat training
(model.load_state_dict(..., strict=True) di evaluate.py), jadi jangan mengubah get_model()
tanpa melatih ulang model.pth.
"""
import torch.nn as nn
from torchvision import models, transforms

ARCH = "convnext_tiny"
IMG_SIZE = 256
NUM_CLASSES = 5
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_model():
    model = models.convnext_tiny(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])