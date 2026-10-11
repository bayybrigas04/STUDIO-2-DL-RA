# DATA READERS 

import os
import random
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as T
from sklearn.model_selection import train_test_split

# 1. KONFIGURASI
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
TRAIN_RATIO = 0.8

DATA_DIR = Path("/content/drive/MyDrive/Datasets/train")

CLASSES = [
    "botol_plastik",
    "gelas_plastik",
    "kertas_kardus",
    "saset_kemasan",
    "styrofoam"
]

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# 2. RANDOM SEED
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 3. CEK DATASET
print("=" * 55)
print("MEMERIKSA DATASET")
print("=" * 55)

if not DATA_DIR.is_dir():
    raise FileNotFoundError(
        f"Folder dataset tidak ditemukan: {DATA_DIR}\n"
        "Pastikan Google Drive sudah di-mount."
    )

for class_name in CLASSES:
    class_path = DATA_DIR / class_name
    if not class_path.is_dir():
        raise FileNotFoundError(
            f"Folder kelas tidak ditemukan: {class_path}"
        )

# 4. TRANSFORMASI GAMBAR
train_transform = T.Compose([
    T.RandomResizedCrop(
        IMG_SIZE, scale=(0.8, 1.0)
    ),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 5. MEMBACA DAN MEMERIKSA GAMBAR
print("\nMemuat gambar...")

# ImageFolder otomatis memberikan label berdasarkan folder kelas.
# Gambar yang rusak diperiksa terlebih dahulu.

valid_extensions = EXTENSIONS
valid_images = []
invalid_images = []

for class_name in CLASSES:
    class_path = DATA_DIR / class_name

    for image_path in sorted(class_path.rglob("*")):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in valid_extensions:
            continue

        try:
            with Image.open(image_path) as img:
                img.verify()
            valid_images.append(image_path)
        except Exception:
            invalid_images.append(image_path)

if not valid_images:
    raise ValueError("Tidak ditemukan gambar valid di dataset.")

print("Gambar valid :", len(valid_images))
print("Gambar rusak :", len(invalid_images))

# 6. MEMBUAT DATASET
# Dataset training dan validation memakai transformasi berbeda.

train_dataset_all = datasets.ImageFolder(
    root=DATA_DIR,
    transform=train_transform
)

val_dataset_all = datasets.ImageFolder(
    root=DATA_DIR,
    transform=val_transform
)

class_names = train_dataset_all.classes

if class_names != CLASSES:
    raise ValueError(
        f"Urutan kelas tidak sesuai.\n"
        f"Seharusnya: {CLASSES}\n"
        f"Ditemukan: {class_names}"
    )

# Gunakan hanya gambar yang lolos pemeriksaan.
valid_image_paths = {str(p.resolve()) for p in valid_images}

valid_indices = [
    i for i, (path, label) in enumerate(train_dataset_all.samples)
    if str(Path(path).resolve()) in valid_image_paths
]

labels = np.array([
    train_dataset_all.targets[i] for i in valid_indices
])

# 7. SPLIT TRAIN DAN VALIDATION
train_indices, val_indices = train_test_split(
    valid_indices,
    test_size=1 - TRAIN_RATIO,
    random_state=SEED,
    stratify=labels
)

train_dataset = Subset(train_dataset_all, train_indices)
val_dataset = Subset(val_dataset_all, val_indices)

# 8. DATA LOADER
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    drop_last=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    drop_last=False
)

# 9. CLASS WEIGHTS
train_labels = np.array([
    train_dataset_all.targets[i] for i in train_indices
])

class_counts = np.bincount(
    train_labels,
    minlength=len(class_names)
)

if np.any(class_counts == 0):
    raise ValueError(
        "Ada kelas yang tidak memiliki gambar pada data training."
    )

weights = len(train_labels) / (
    len(class_names) * class_counts
)

class_weights = torch.tensor(
    weights, dtype=torch.float32
)

# 10. HASIL PEMBACAAN DATA
print("\n" + "=" * 55)
print("DATA READERS BERHASIL")
print("=" * 55)

print("Lokasi dataset :", DATA_DIR)
print("Nama kelas     :", class_names)
print("Total gambar   :", len(valid_indices))
print("Data training  :", len(train_dataset))
print("Data validation:", len(val_dataset))
print("Batch training :", len(train_loader))
print("Batch validasi :", len(val_loader))
print("\nJumlah gambar per kelas pada training:")

for i, name in enumerate(class_names):
    print(f"{name}: {class_counts[i]} gambar")

print("\nClass weights:", class_weights)
print("Device:", "CUDA" if torch.cuda.is_available() else "CPU")

# 11. UJI SATU BATCH
images, labels = next(iter(train_loader))

print("\nPemeriksaan batch:")
print("Bentuk gambar:", images.shape)
print("Bentuk label :", labels.shape)
print("Label contoh :", labels[:10].tolist())
print("\nSemua data siap digunakan untuk training!")
