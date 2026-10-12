import argparse
import importlib.util
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

from submission.model import IMG_SIZE, MEAN, STD, get_transform

CLASSES = ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"]
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

HPARAMS = {
    "convnext_tiny":   dict(lr=1e-4, wd=0.05),
    "resnet50":        dict(lr=2e-4, wd=1e-2),
    "efficientnet_b0": dict(lr=3e-4, wd=1e-2),
}


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
    ch, cw_ = int(H * r), int(W * r)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
    x1, x2 = max(cx - cw_ // 2, 0), min(cx + cw_ // 2, W)
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1.0 - ((y2 - y1) * (x2 - x1) / (H * W))
    return x, y, y[idx], lam


def create_model(arch: str, num_classes: int = len(CLASSES), pretrained: bool = True):
    """Membangun arsitektur dan memuat bobot pretrained ImageNet (hanya untuk training)."""
    model_fn = getattr(models, arch)
    model = model_fn(weights="DEFAULT" if pretrained else None)
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Head arsitektur {arch} tidak dikenali.")
    return model


def get_head_parameters(model):
    if hasattr(model, "fc"):
        return model.fc.parameters()
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            return model.classifier[-1].parameters()
        return model.classifier.parameters()
    raise ValueError("Head layer tidak ditemukan.")


def scan_dataset(data_dir: Path):
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
    """Stratified Group split bila CSV tersedia (gambar dari halaman web yang sama tidak
    terpisah ke train dan val), atau stratified acak bila tidak."""
    csv_file = Path(csv_path) if csv_path else None
    use_csv = False
    if csv_file and csv_file.is_file():
        src = pd.read_csv(csv_file)
        # [REVISI] jangan crash bila kolom CSV berbeda; turun ke split acak dengan peringatan
        if {"file", "page_url"} <= set(src.columns):
            src["name"] = src["file"].astype(str).str.split("/").str[-1]
            df = df.merge(src[["name", "page_url"]].drop_duplicates("name"), on="name", how="left")
            df["group"] = df["page_url"].fillna(df["name"])
            print(f"Group split dari CSV: {csv_file} (cocok: {df['page_url'].notna().sum()}/{len(df)})")
            use_csv = True
        else:
            print(f"[PERINGATAN] CSV {csv_file} tidak punya kolom 'file' dan 'page_url' (kolom: {list(src.columns)})")
    else:
        print(f"[PERINGATAN] CSV tidak ditemukan ({csv_path}).")
    if not use_csv:
        df["group"] = df["name"]
        print("-> memakai stratified split acak (tanpa perlindungan kebocoran antar-halaman).")

    y_all = df["label"].map({c: i for i, c in enumerate(CLASSES)}).values
    if df["group"].nunique() < len(df):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        tr_idx, va_idx = next(sgkf.split(df, y_all, groups=df["group"]))
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1.0 / n_splits, random_state=seed)
        tr_idx, va_idx = next(sss.split(df, y_all))
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
        target = self.label_map[row["label"]]
        if self.transform:
            image = self.transform(image)
        return image, target


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
    labels = list(range(len(CLASSES)))
    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    acc = float((np.array(preds) == np.array(targets)).mean())
    per_cls = f1_score(targets, preds, average=None, labels=labels, zero_division=0)
    per_cls = {c: round(float(f), 4) for c, f in zip(CLASSES, per_cls)}
    cm = confusion_matrix(targets, preds, labels=labels).tolist()  # [REVISI] untuk analisis error
    return f1, acc, per_cls, cm


def export_val(val_df: pd.DataFrame, dst):
    """[REVISI] Salin gambar validasi ke dst/<kelas>/ agar bisa diuji dengan evaluate.py."""
    dst = Path(dst)
    if dst.exists() and any(dst.iterdir()):
        print(f"[export] {dst} sudah ada, dilewati. Hapus foldernya bila split berubah.")
        return
    for _, r in val_df.iterrows():
        out = dst / r["label"]
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r["path"], out / r["name"])
    print(f"[export] {len(val_df)} gambar validasi disalin ke {dst}")


def check_submission(state, sub_dir):
    """[REVISI] Cek apakah bobot cocok dengan get_model() di submission/model.py
    (cara yang sama seperti evaluate.py: load_state_dict strict)."""
    model_py = Path(sub_dir) / "model.py"
    if not model_py.is_file():
        return False, f"{model_py} tidak ditemukan"
    spec = importlib.util.spec_from_file_location("submission_model_check", model_py)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(sub_dir).resolve()))
    spec.loader.exec_module(mod)
    try:
        mod.get_model().load_state_dict(state, strict=True)
    except RuntimeError as e:
        return False, str(e).splitlines()[0]
    return True, "OK"


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    # [REVISI] ukuran gambar harus sama dengan submission/model.py
    if args.img_size not in (None, IMG_SIZE):
        raise SystemExit(f"[GAGAL] --img-size {args.img_size} != IMG_SIZE={IMG_SIZE} di submission/model.py. "
                         "Ubah IMG_SIZE dan get_transform() di model.py, jangan lewat argumen.")
    img_size = IMG_SIZE

    data_dir = Path(args.data).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = scan_dataset(data_dir)
    train_df, val_df = split_dataset(df, csv_path=args.csv, seed=args.seed)
    print(f"Train data: {len(train_df)} | Val data: {len(val_df)}")
    if args.export_val:
        export_val(val_df, args.export_val)

    train_tf = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.4, 1.0), ratio=(0.75, 1.333)),
        T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.2), value="random"),
    ])
    val_tf = get_transform()  # [REVISI] persis transform yang dipakai evaluate.py

    train_ds = CustomImageDataset(train_df, transform=train_tf)
    val_ds = CustomImageDataset(val_df, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=use_amp, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=use_amp)

    targets = [train_ds.label_map[l] for l in train_df["label"]]
    counts = np.bincount(targets, minlength=len(CLASSES))
    cw = (counts.sum() / counts) ** 0.5
    class_weights = torch.tensor(cw / cw.mean(), dtype=torch.float32).to(device)

    hp = HPARAMS.get(args.arch, dict(lr=1e-4, wd=0.05))
    lr = args.lr if args.lr is not None else hp["lr"]            # [REVISI] default None, bukan 1e-4
    wd = args.weight_decay if args.weight_decay is not None else hp["wd"]

    print(f"\nMembangun model: {args.arch} (pretrained, lr={lr}, wd={wd}, img={img_size})...")
    model = create_model(args.arch, num_classes=len(CLASSES), pretrained=True).to(device)

    head_params = list(get_head_parameters(model))
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr},
        {"params": head_params, "lr": lr * args.head_lr_mult},
    ], weight_decay=wd)

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

    print("\n" + "=" * 60)
    print(f"MULAI PELATIHAN ({args.epochs} Epochs)")
    print("=" * 60)

    best_f1, best_epoch, best_state, best_per_cls, best_cm, history = -1.0, 0, None, None, None, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            use_cm = (args.cutmix_prob > 0) and (random.random() < args.cutmix_prob)
            if use_cm:
                x, ya, yb, lam = cutmix(x, y)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = (lam * criterion(out, ya) + (1.0 - lam) * criterion(out, yb)) if use_cm \
                    else criterion(out, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()

        epoch_loss = running_loss / max(1, len(train_loader))
        val_f1, val_acc, per_cls, cm = evaluate_loader(model, val_loader, device, use_amp=use_amp)
        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_macro_f1": val_f1, "val_acc": val_acc})
        mark = ""
        if val_f1 > best_f1:
            best_f1, best_epoch, best_per_cls, best_cm = val_f1, epoch, per_cls, cm
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            mark = " [BEST]"
        print(f"Epoch {epoch:2d}/{args.epochs} | Loss: {epoch_loss:.4f} | Val Macro-F1: {val_f1:.4f} | "
              f"Val Acc: {val_acc:.4f} | {time.time() - t_ep:.0f} dtk{mark}")

    total_min = (time.time() - t0) / 60
    print("\n" + "=" * 60)
    print(f"PELATIHAN SELESAI ({total_min:.1f} menit) | Best Macro-F1: {best_f1:.4f} (epoch {best_epoch})")
    print("=" * 60)
    # [REVISI] laporan per kelas dari epoch terbaik
    for c, f in best_per_cls.items():
        print(f"  F1 {c:<15} {f:.4f}")
    print("Confusion matrix (baris = label asli, kolom = prediksi):")
    for c, row in zip(CLASSES, best_cm):
        print(f"  {c:<15} {row}")

    # Simpan checkpoint (FP16 bila > 95 MB agar muat di GitHub)
    fp32_mb = sum(v.numel() * 4 for v in best_state.values() if v.is_floating_point()) / 1e6
    save_state = best_state
    if fp32_mb > 95:
        print(f"Model {fp32_mb:.1f} MB (> 95 MB) -> disimpan dalam FP16...")
        save_state = {k: (v.half() if v.is_floating_point() else v) for k, v in best_state.items()}

    # [REVISI] model.pth hanya ditimpa bila cocok dengan submission/model.py
    ok, msg = check_submission(save_state, out_dir)
    if ok:
        target = out_dir / "model.pth"
    else:
        Path("experiments").mkdir(exist_ok=True)
        target = Path("experiments") / f"{args.arch}.pth"
        print(f"\n[PERINGATAN] Bobot {args.arch} TIDAK cocok dengan {out_dir / 'model.py'}:\n  {msg}")
        print(f"  -> disimpan ke {target}; submission/model.pth TIDAK diubah.")
        print("  Untuk memakainya sebagai submission, ubah get_model() di model.py ke arsitektur yang sama.")
    torch.save(save_state, target)
    print(f"Bobot disimpan ke: {target} ({target.stat().st_size / 1e6:.1f} MB)")

    Path("results").mkdir(exist_ok=True)
    Path(f"results/{args.arch}.json").write_text(json.dumps({
        "arch": args.arch, "best_val_macro_f1": best_f1, "best_epoch": best_epoch,
        "per_class_f1": best_per_cls, "confusion_matrix": best_cm, "img_size": img_size,
        "lr": lr, "weight_decay": wd, "saved_to": str(target), "minutes": round(total_min, 1),
        "history": history}, indent=2))


def main():
    p = argparse.ArgumentParser(description="Training Pipeline Studio 2")
    p.add_argument("--data", default="./train", help="Folder dataset train/")
    p.add_argument("--csv", default="./train_sources.csv", help="File train_sources.csv")
    p.add_argument("--arch", default="convnext_tiny", choices=list(HPARAMS), help="Backbone")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="lr backbone (default: sesuai HPARAMS)")
    p.add_argument("--head-lr-mult", type=float, default=10.0)
    p.add_argument("--weight-decay", type=float, default=None, help="default: sesuai HPARAMS")
    p.add_argument("--img-size", type=int, default=None, help="harus sama dengan IMG_SIZE di submission/model.py")
    p.add_argument("--cutmix-prob", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--export-val", help="salin gambar validasi ke folder ini untuk evaluate.py")
    p.add_argument("--out-dir", default="./submission", help="Folder submission (model.py dan model.pth)")
    train(p.parse_args())


if __name__ == "__main__":
    main()