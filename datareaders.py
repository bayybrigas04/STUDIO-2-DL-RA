import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from submission.model import IMG_SIZE, MEAN, STD, get_transform

CLASSES = ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SEED = 42
SPLIT_SEED = 42  # terpisah dari SEED agar split validasi sama untuk semua seed
BATCH_SIZE = 32
NUM_WORKERS = 2
N_FOLDS = 7


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_train_transform(crop_min=0.4, erase_p=0.25):
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(crop_min, 1.0), ratio=(0.75, 1.333)),
        T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=erase_p, scale=(0.02, 0.2), value="random"),
    ])


class WasteDataset(Dataset):
    def __init__(self, frame, transform=None):
        self.paths = frame["path"].tolist()
        self.labels = frame["label"].map(CLASS_TO_IDX).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index]


def scan_dataset(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {data_dir}")

    records, n_bad = [], 0
    for cls in CLASSES:
        class_dir = data_dir / cls
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Folder kelas tidak ditemukan: {class_dir}")
        for p in sorted(class_dir.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in EXTENSIONS:
                continue
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                n_bad += 1
                continue
            records.append((cls, p.name, str(p.resolve())))

    if not records:
        raise ValueError(f"Tidak ada gambar valid di {data_dir}")
    frame = pd.DataFrame(records, columns=["label", "name", "path"])
    print(f"Gambar valid: {len(frame)} | rusak: {n_bad}")
    return frame


def split_dataset(frame, csv_path=None, seed=SPLIT_SEED, n_folds=N_FOLDS):
    frame = frame.copy()
    csv_file = Path(csv_path) if csv_path else None

    if csv_file and csv_file.is_file():
        source = pd.read_csv(csv_file)
        if not {"file", "page_url"}.issubset(source.columns):
            raise ValueError("CSV sumber harus memiliki kolom 'file' dan 'page_url'.")
        source["name"] = source["file"].astype(str).str.split("/").str[-1]
        source = source[["name", "page_url"]].drop_duplicates("name")
        frame = frame.merge(source, on="name", how="left")
        frame["group"] = frame["page_url"].fillna(frame["name"])
        print(f"CSV sumber: {csv_file} | cocok: {frame['page_url'].notna().sum()}/{len(frame)}")
    else:
        frame["group"] = frame["name"]
        print("train_sources.csv tidak ditemukan; memakai stratified split berdasarkan nama file.")

    y = frame["label"].map(CLASS_TO_IDX).to_numpy()
    if frame["group"].nunique() < len(frame):
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        train_idx, val_idx = next(splitter.split(frame, y, groups=frame["group"]))
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=1.0 / n_folds, random_state=seed)
        train_idx, val_idx = next(splitter.split(frame, y))

    extra = ["page_url", "group"]
    train_frame = frame.iloc[train_idx].drop(columns=extra, errors="ignore").reset_index(drop=True)
    val_frame = frame.iloc[val_idx].drop(columns=extra, errors="ignore").reset_index(drop=True)
    return train_frame, val_frame


def class_weights_from(frame):
    """Bobot kelas akar-terbalik, dinormalisasi ke rata-rata 1."""
    labels = frame["label"].map(CLASS_TO_IDX).to_numpy()
    counts = np.bincount(labels, minlength=len(CLASSES))
    if np.any(counts == 0):
        raise ValueError("Ada kelas yang tidak memiliki gambar pada data training.")
    weights = np.sqrt(counts.sum() / counts)
    return counts, weights / weights.mean()


def get_data(data_dir, csv_path=None, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
             seed=SEED, split_seed=SPLIT_SEED, n_folds=N_FOLDS, crop_min=0.4, erase_p=0.25,
             all_data=False, verbose=True):
    set_seed(seed)
    frame = scan_dataset(data_dir)
    train_frame, val_frame = split_dataset(frame, csv_path=csv_path, seed=split_seed, n_folds=n_folds)
    if all_data:
        train_frame = pd.concat([train_frame, val_frame]).reset_index(drop=True)
        if verbose:
            print(f"[ALL-DATA] validasi ikut dilatih -> train {len(train_frame)} gambar "
                  f"(skor validasi tidak lagi independen)")

    train_ds = WasteDataset(train_frame, get_train_transform(crop_min, erase_p))
    val_ds = WasteDataset(val_frame, get_transform())

    loader_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    counts, weights = class_weights_from(train_frame)

    if verbose:
        print(f"Train: {len(train_ds)} | Validasi: {len(val_ds)}")
        for i, name in enumerate(CLASSES):
            print(f"  {name:<15} train={counts[i]}")
        print("Class weights:", [round(float(w), 3) for w in weights])

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "class_weights": torch.tensor(weights, dtype=torch.float32),
        "class_counts": counts,
        "val_samples": list(zip(val_frame["path"].tolist(), val_ds.labels)),
        "train_frame": train_frame,
        "val_frame": val_frame,
    }


def export_val_folder(val_samples, dst):
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    for path, label in val_samples:
        out_dir = dst / CLASSES[label]
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out_dir / Path(path).name)
    print(f"Validasi diekspor: {len(val_samples)} gambar -> {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Folder dataset dengan 5 subfolder kelas")
    parser.add_argument("--csv", default="./train_sources.csv", help="CSV sumber gambar (opsional)")
    parser.add_argument("--export-val", help="Folder tujuan ekspor validasi")
    parser.add_argument("--all-data", action="store_true", help="Gabungkan validasi ke train")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    data = get_data(args.data, csv_path=args.csv, num_workers=args.workers, all_data=args.all_data)
    x, y = next(iter(data["train_loader"]))
    print("Batch train:", tuple(x.shape), "| label contoh:", y[:8].tolist())
    x, y = next(iter(data["val_loader"]))
    print("Batch val  :", tuple(x.shape), "| label contoh:", y[:8].tolist())
    if args.export_val:
        export_val_folder(data["val_samples"], args.export_val)
    print("OK: datareaders berfungsi")