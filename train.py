import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torchvision import models
from torchvision import transforms as T

from datareaders import get_data, export_val_folder
from submission.model import IMG_SIZE, MEAN, STD, get_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pretrained_backbone(model):
    """Isi bobot ImageNet yang cocok; classifier 5 kelas tetap diinisialisasi baru."""
    pretrained = models.convnext_tiny(weights="DEFAULT").state_dict()
    own = model.state_dict()
    compatible = {k: v for k, v in pretrained.items()
                  if k in own and v.shape == own[k].shape}
    result = model.load_state_dict(compatible, strict=False)
    print("Bobot ImageNet dimuat.")
    if result.missing_keys:
        print("Parameter baru/tidak dimuat (normalnya classifier):", result.missing_keys)
    if result.unexpected_keys:
        print("Parameter pretrained yang tidak digunakan:", result.unexpected_keys)


def build_train_transform(crop_min, erase_p):
    """Augmentasi khusus training. Ukuran output & normalisasi sama dengan submission/model.py."""
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(crop_min, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.TrivialAugmentWide(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=erase_p),
    ])


def cutmix(images, labels):
    lam = np.random.beta(1.0, 1.0)
    index = torch.randperm(images.size(0), device=images.device)
    height, width = images.shape[2:]
    ratio = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(height * ratio), int(width * ratio)
    cy, cx = np.random.randint(height), np.random.randint(width)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, height)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, width)
    images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    lam = 1.0 - ((y2 - y1) * (x2 - x1) / (height * width))
    return images, labels, labels[index], lam


@torch.no_grad()
def evaluate_loader(model, loader, device, use_amp):
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(labels.tolist())
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = float((np.asarray(y_true) == np.asarray(y_pred)).mean())
    return macro_f1, accuracy


def verify_checkpoint(model_path):
    """Pastikan model.pth cocok dengan submission/model.py (strict=True, seperti evaluate.py)."""
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    get_model().load_state_dict(state, strict=True)
    print("Verifikasi OK: model.pth cocok dengan submission/model.py (strict=True).")


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    output_dir = Path(args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = get_data(
        args.data, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed,
        csv_path=args.csv, group_col=args.group_col,
    )
    train_loader, val_loader = data["train_loader"], data["val_loader"]
    class_weights = data["class_weights"].to(device)

    # get_data() memakai augmentasi sederhana untuk train; ganti dengan augmentasi
    # kita. Train & validasi adalah dua ImageFolder terpisah, jadi validasi tidak terpengaruh.
    train_loader.dataset.dataset.transform = build_train_transform(args.crop_min, args.erase_p)

    model = get_model()
    if args.pretrained:
        load_pretrained_backbone(model)
    model = model.to(device)

    head = model.classifier[-1]
    head_param_ids = {id(p) for p in head.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr},
            {"params": head.parameters(), "lr": args.lr * args.head_lr_mult},
        ],
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError("DataLoader train kosong. Coba turunkan batch size atau periksa dataset.")
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights if args.class_weights else None,
        label_smoothing=args.label_smoothing,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1, best_epoch, best_state = -1.0, 0, None
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            use_cutmix = args.cutmix_prob > 0 and random.random() < args.cutmix_prob
            if use_cutmix:
                images, labels_a, labels_b, lam = cutmix(images, labels)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                if use_cutmix:
                    loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
                else:
                    loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()

        val_f1, val_acc = evaluate_loader(model, val_loader, device, use_amp)
        epoch_loss = running_loss / max(1, len(train_loader))
        marker = ""
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            marker = " [BEST]"
        print(
            f"Epoch {epoch:02d}/{args.epochs} | loss={epoch_loss:.4f} "
            f"| val_macro_f1={val_f1:.4f} | val_acc={val_acc:.4f}{marker}"
        )

    if best_state is None:
        raise RuntimeError("Training tidak menghasilkan checkpoint.")

    model_path = output_dir / "model.pth"
    fp32_mb = sum(v.numel() * 4 for v in best_state.values() if v.is_floating_point()) / 1e6
    save_state = best_state
    if fp32_mb > 95:
        print(f"Ukuran FP32 {fp32_mb:.1f} MB > 95 MB -> disimpan sebagai FP16.")
        save_state = {k: (v.half() if v.is_floating_point() else v)
                      for k, v in best_state.items()}
    torch.save(save_state, model_path)
    elapsed = (time.time() - start) / 60
    print(f"\nBest Macro-F1: {best_f1:.4f} | epoch: {best_epoch}")
    print(f"Waktu training: {elapsed:.1f} menit")
    print(f"Checkpoint: {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")

    verify_checkpoint(model_path)

    if args.export_val:
        export_val_folder(data["val_samples"], args.export_val)
    print("\nTraining selesai. Folder submission berisi model.py dan model.pth.")


def main():
    parser = argparse.ArgumentParser(description="Training ConvNeXt-Tiny Studio 2")
    parser.add_argument("--data", default="./train", help="Dataset dengan lima folder kelas")
    parser.add_argument("--csv", default="", help="train_sources.csv untuk group split; kosong = split acak")
    parser.add_argument("--group-col", default="page_url", help="Kolom CSV yang dipakai sebagai grup sumber")
    parser.add_argument("--out-dir", default="./submission", help="Folder output berisi model.pth")
    parser.add_argument("--export-val", default="./my_val", help="Ekspor data validasi untuk evaluate.py; kosongkan untuk menonaktifkan")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr-mult", type=float, default=10.0)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE, help="Harus cocok dengan submission/model.py")
    parser.add_argument("--cutmix-prob", type=float, default=0.5)
    parser.add_argument("--crop-min", type=float, default=0.4, help="Skala minimum RandomResizedCrop")
    parser.add_argument("--erase-p", type=float, default=0.25, help="Probabilitas RandomErasing")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    # IMG_SIZE ditetapkan di submission/model.py agar transform train/validasi/evaluate konsisten.
    if args.img_size != IMG_SIZE:
        parser.error(f"--img-size={args.img_size} tidak cocok dengan submission/model.py IMG_SIZE={IMG_SIZE}")
    if args.csv == "":
        args.csv = None
    if args.export_val == "":
        args.export_val = None
    train(args)


if __name__ == "__main__":
    main()