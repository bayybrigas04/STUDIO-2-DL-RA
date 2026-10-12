import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as T

# Validasi memakai get_transform() dari submission/model.py supaya preprocessing
# saat training SAMA dengan saat evaluate.py menilai model.
from submission.model import IMG_SIZE, MEAN, STD, get_transform

SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 2
VAL_RATIO = 0.2  # 80% train, 20% validasi

CLASSES = ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_train_transform():
    """Augmentasi untuk data training saja."""
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def check_images(data_dir):
    """Periksa folder kelas dan gambar. Kembalikan (path_valid, path_rusak)."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {data_dir}")
    valid, invalid = [], []
    for name in CLASSES:
        class_dir = data_dir / name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Folder kelas tidak ditemukan: {class_dir}")
        for p in sorted(class_dir.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in EXTENSIONS:
                continue
            try:
                with Image.open(p) as img:
                    img.verify()
                valid.append(p)
            except Exception:
                invalid.append(p)
    if not valid:
        raise ValueError("Tidak ditemukan gambar valid di dataset.")
    return valid, invalid


def get_data(data_dir, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
             val_ratio=VAL_RATIO, seed=SEED, verbose=True):
    """Bangun DataLoader train & validasi dari folder berisi 5 folder kelas."""
    set_seed(seed)
    valid, invalid = check_images(data_dir)
    valid_set = {str(p.resolve()) for p in valid}
    is_valid = lambda path: str(Path(path).resolve()) in valid_set  # buang gambar rusak

    # Dua dataset di folder yang sama, beda transform (agar validasi tidak diaugmentasi)
    train_all = datasets.ImageFolder(data_dir, get_train_transform(), is_valid_file=is_valid)
    val_all = datasets.ImageFolder(data_dir, get_transform(), is_valid_file=is_valid)

    if train_all.classes != CLASSES:
        raise ValueError(f"Urutan kelas tidak sesuai.\nSeharusnya: {CLASSES}\nDitemukan: {train_all.classes}")

    targets = np.array(train_all.targets)
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)), test_size=val_ratio,
        random_state=seed, stratify=targets)

    train_ds, val_ds = Subset(train_all, train_idx), Subset(val_all, val_idx)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)

    counts = np.bincount(targets[train_idx], minlength=len(CLASSES))
    if np.any(counts == 0):
        raise ValueError("Ada kelas tanpa gambar pada data training.")
    weights = len(train_idx) / (len(CLASSES) * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    if verbose:
        print(f"Gambar valid: {len(valid)} | rusak: {len(invalid)}")
        print(f"Train: {len(train_ds)} | Validasi: {len(val_ds)}")
        for i, name in enumerate(CLASSES):
            print(f"  {name:<15} train={counts[i]}")
        print("Class weights:", [round(float(w), 3) for w in weights])

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "class_weights": class_weights,
        "class_counts": counts,
        "val_samples": [val_all.samples[i] for i in val_idx],  # [(path, label), ...]
    }


def export_val_folder(val_samples, dst):
    """Salin gambar validasi ke dst/<kelas>/ agar bisa diuji dengan evaluate.py:
        python evaluate.py --submission ./submission --data <dst>
    """
    dst = Path(dst)
    if dst.exists():
        print(f"[export] {dst} sudah ada, dilewati")
        return
    for path, label in val_samples:
        out = dst / CLASSES[label]
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out / Path(path).name)
    print(f"[export] {len(val_samples)} gambar validasi disalin ke {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder train/ (5 folder kelas)")
    ap.add_argument("--export-val", help="salin gambar validasi ke folder ini untuk evaluate.py")
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    a = ap.parse_args()

    data = get_data(a.data, num_workers=a.workers)
    images, labels = next(iter(data["train_loader"]))
    print("Batch train:", tuple(images.shape), "label:", labels[:8].tolist())
    images, labels = next(iter(data["val_loader"]))
    print("Batch val  :", tuple(images.shape), "label:", labels[:8].tolist())
    if a.export_val:
        export_val_folder(data["val_samples"], a.export_val)
    print("OK: datareaders berfungsi")
