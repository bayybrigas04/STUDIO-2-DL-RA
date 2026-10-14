"""Skrip Pelatihan Studio 2 — Klasifikasi Sampah (Khusus ResNet-50).

Pipeline:
  - Arsitektur: ResNet-50 pretrained ImageNet
  - Stratified Group Split berbasis train_sources.csv
  - Augmentasi: TrivialAugmentWide + RandomResizedCrop + RandomErasing + CutMix
  - AdamW (lr=2e-4, wd=0.01) + Warmup (1 epoch) + Cosine Annealing
  - Differential LR: Head (model.fc) 10x lebih besar dari backbone
  - Tracking Macro-F1 & otomatis simpan FP16 (< 60 MB) ke submission/model.pth
"""
import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

CLASSES = ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"]
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cutmix(x, y):
    lam = np.random.beta(1.0, 1.0)
    idx = torch.randperm(x.size(0), device=x.device)
    H, W = x.shape[2:]
    r = math.sqrt(1.0 - lam)
    ch, cw = int(H * r), int(W * r)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
    x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1.0 - ((y2 - y1) * (x2 - x1) / (H * W))
    return x, y, y[idx], lam


def create_model(num_classes: int = len(CLASSES), pretrained: bool = True):
    """Membangun ResNet-50 dan mengganti classification head (fc)."""
    model = models.resnet50(weights="DEFAULT" if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def scan_dataset(data_dir: Path):
    """Memindai dataset dan memvalidasi file gambar."""
    records, bad = [], []
    for cls in CLASSES:
        cls_dir = data_dir / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Folder kelas tidak ditemukan: {cls_dir}")
        for p in sorted(cls_dir.rglob("*")):
            if p.suffix.lower() not in EXTS:
                continue
            try:
                with Image.open(p) as im:
                    im.verify()
                records.append((cls, p.name, str(p.resolve())))
            except Exception as e:
                bad.append((p.name, str(e)))

    if not records:
        raise ValueError(f"Tidak ada gambar valid di folder: {data_dir}")

    df = pd.DataFrame(records, columns=["label", "name", "path"])
    print(f"Total gambar valid: {len(df)} | Rusak (dilewati): {len(bad)}")
    return df


def split_dataset(df: pd.DataFrame, csv_path: str = None, seed: int = 42, n_splits: int = 7):
    """Split train/val dengan Stratified Group jika CSV tersedia, atau Stratified acak."""
    csv_file = Path(csv_path) if csv_path else None
    if csv_file and csv_file.is_file():
        src = pd.read_csv(csv_file)
        src["name"] = src["file"].astype(str).str.split("/").str[-1]
        df = df.merge(src[["name", "page_url"]].drop_duplicates("name"), on="name", how="left")
        df["group"] = df["page_url"].fillna(df["name"])
        matched = df["page_url"].notna().sum()
        print(f"Menggunakan Group K-Fold dari CSV: {csv_file} (Cocok: {matched}/{len(df)})")
    else:
        df["group"] = df["name"]
        print("CSV tidak ditemukan -> Menggunakan stratified split acak.")

    y = df["label"].map({c: i for i, c in enumerate(CLASSES)}).values
    if df["group"].nunique() < len(df):
        tr_idx, va_idx = next(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(df, y, df["group"]))
    else:
        tr_idx, va_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=1.0 / n_splits, random_state=seed).split(df, y))

    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)


class CustomImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = {c: i for i, c in enumerate(CLASSES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(row["path"]) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label_map[row["label"]]


@torch.no_grad()
def evaluate_loader(model, loader, device, use_amp=True):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        preds.extend(logits.argmax(1).cpu().tolist())
        targets.extend(y.tolist())

    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    acc = (np.array(preds) == np.array(targets)).mean()
    return f1, acc


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    print(f"Device: {device} | AMP: {use_amp}")

    data_dir = Path(args.data).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pindai dan split dataset
    df = scan_dataset(data_dir)
    train_df, val_df = split_dataset(df, csv_path=args.csv, seed=args.seed)
    print(f"Train data: {len(train_df)} | Val data: {len(val_df)}")

    # 2. Transformasi gambar
    train_tf = T.Compose([
        T.RandomResizedCrop(args.img_size, scale=(0.4, 1.0), ratio=(0.75, 1.333)),
        T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.2), value="random"),
    ])

    val_tf = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

    train_ds = CustomImageDataset(train_df, transform=train_tf)
    val_ds = CustomImageDataset(val_df, transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=use_amp, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_amp
    )

    # Class weights
    targets = [train_ds.label_map[l] for l in train_df["label"]]
    counts = np.bincount(targets, minlength=len(CLASSES))
    cw = (counts.sum() / counts) ** 0.5
    class_weights = torch.tensor(cw / cw.mean(), dtype=torch.float32).to(device)

    # 3. Model ResNet-50 & Optimizer
    print(f"\nMembangun model: ResNet-50 (lr: {args.lr}, wd: {args.weight_decay})...")
    model = create_model(num_classes=len(CLASSES), pretrained=True).to(device)

    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc")]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr},
        {"params": model.fc.parameters(), "lr": args.lr * args.head_lr_mult}
    ], weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 4. Training Loop
    print("\n" + "=" * 60)
    print(f"MULAI PELATIHAN ({args.epochs} Epochs)")
    print("=" * 60)

    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            use_cm = (args.cutmix_prob > 0) and (random.random() < args.cutmix_prob)
            if use_cm:
                x, ya, yb, lam = cutmix(x, y)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = lam * criterion(out, ya) + (1.0 - lam) * criterion(out, yb) if use_cm else criterion(out, y)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()

        epoch_loss = running_loss / max(1, len(train_loader))
        val_f1, val_acc = evaluate_loader(model, val_loader, device, use_amp=use_amp)

        mark = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            mark = " [BEST]"

        print(f"Epoch {epoch:2d}/{args.epochs} | Loss: {epoch_loss:.4f} | Val Macro-F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}{mark}")

    total_time = (time.time() - t0) / 60
    print("\n" + "=" * 60)
    print(f"PELATIHAN SELESAI ({total_time:.1f} menit)")
    print(f"Best Macro-F1: {best_f1:.4f} pada Epoch {best_epoch}")
    print("=" * 60)

    # 5. Simpan Checkpoint (otomatis FP16 jika > 95 MB agar < 60 MB)
    model_pth = out_dir / "model.pth"
    fp32_mb = sum(v.numel() * 4 for v in best_state.values() if v.is_floating_point()) / 1e6
    save_state = {k: (v.half() if v.is_floating_point() else v) for k, v in best_state.items()} if fp32_mb > 95 else best_state
    torch.save(save_state, model_pth)
    print(f"Bobot terbaik disimpan ke: {model_pth} ({model_pth.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Training ResNet-50 Studio 2")
    parser.add_argument("--data", default="./train", help="Folder dataset train/")
    parser.add_argument("--csv", default="./train_sources.csv", help="File train_sources.csv")
    parser.add_argument("--epochs", type=int, default=15, help="Jumlah epoch")
    parser.add_argument("--batch-size", type=int, default=32, help="Ukuran batch")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate backbone")
    parser.add_argument("--head-lr-mult", type=float, default=10.0, help="Pengali lr untuk head")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--img-size", type=int, default=256, help="Resolusi gambar (256)")
    parser.add_argument("--cutmix-prob", type=float, default=0.5, help="Peluang CutMix")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out-dir", default="./submission", help="Folder output model.pth")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
