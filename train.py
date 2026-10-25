import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from datareaders import get_data, set_seed
from submission.model import get_model

CHANNELS_LAST = torch.channels_last


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
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    return x, y, y[idx], lam


class EMA:
    """EMA bobot. `shadow` adalah model utuh, langsung bisa dipakai untuk evaluasi."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        torch._foreach_mul_(list(self.shadow.parameters()), self.decay)
        torch._foreach_add_(list(self.shadow.parameters()), list(model.parameters()),
                            alpha=1.0 - self.decay)


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True, memory_format=CHANNELS_LAST)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            preds.append(model(x).argmax(1).cpu())
        targets.append(y)
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    return f1, float((preds == targets).mean())


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = get_data(
        args.data, csv_path=args.csv, batch_size=args.batch_size, num_workers=args.num_workers,
        seed=args.seed, split_seed=args.split_seed, crop_min=args.crop_min, erase_p=args.erase_p,
        all_data=args.all_data,
    )
    train_loader, val_loader = data["train_loader"], data["val_loader"]
    criterion = nn.CrossEntropyLoss(
        weight=data["class_weights"].to(device), label_smoothing=args.label_smoothing
    )

    # init_seed hanya mengubah inisialisasi head; RNG data/augmentasi dikembalikan ke --seed.
    if args.init_seed >= 0:
        torch.manual_seed(args.init_seed)
    model = get_model(pretrained=not args.no_pretrained, drop_path=args.drop_path)
    model = model.to(device, memory_format=CHANNELS_LAST)
    if args.init_seed >= 0:
        set_seed(args.seed)

    ema = EMA(model, args.ema) if args.ema > 0 else None
    eval_model = ema.shadow if ema else model

    head = model.classifier[-1]
    head_ids = {id(p) for p in head.parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": args.lr},
         {"params": list(head.parameters()), "lr": args.lr * args.head_lr_mult}],
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    keep_last = args.all_data or args.last_epoch
    best_f1, best_epoch, best_state, history = -1.0, 0, None, []
    t0 = time.time()
    print(f"\nMULAI PELATIHAN ({args.epochs} epoch, lr={args.lr}, wd={args.weight_decay})")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = torch.zeros((), device=device)

        for x, y in train_loader:
            x = x.to(device, non_blocking=True, memory_format=CHANNELS_LAST)
            y = y.to(device, non_blocking=True)
            use_cm = args.cutmix_prob > 0 and random.random() < args.cutmix_prob
            if use_cm:
                x, ya, yb, lam = cutmix(x, y)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                if use_cm:
                    loss = lam * criterion(out, ya) + (1.0 - lam) * criterion(out, yb)
                else:
                    loss = criterion(out, y)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema:
                ema.update(model)
            running_loss += loss.detach()

        epoch_loss = running_loss.item() / steps_per_epoch
        val_f1, val_acc = evaluate(eval_model, val_loader, device, use_amp)
        history.append((epoch, epoch_loss, float(val_f1), float(val_acc)))

        if keep_last:
            save_now = epoch == args.epochs
        else:
            save_now = val_f1 > best_f1
        if save_now:
            best_f1, best_epoch = val_f1, epoch
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in eval_model.state_dict().items()}

        print(f"Epoch {epoch:2d}/{args.epochs} | Loss: {epoch_loss:.4f} | "
              f"Val Macro-F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}{' [SAVED]' if save_now else ''}")

    minutes = (time.time() - t0) / 60
    print(f"\nSELESAI ({minutes:.1f} menit) | Macro-F1 tersimpan: {best_f1:.4f} (epoch {best_epoch})")

    model_pth = out_dir / "model.pth"
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in best_state.items()}, model_pth)
    size_mb = model_pth.stat().st_size / 1e6
    print(f"Bobot disimpan ke: {model_pth} ({size_mb:.1f} MB)")

    result = {
        "args": vars(args), "best_f1": float(best_f1), "best_epoch": best_epoch,
        "last3_f1": float(np.mean([h[2] for h in history[-3:]])),
        "minutes": round(minutes, 1), "size_mb": round(size_mb, 1), "history": history,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="Training ConvNeXt-Tiny Studio 2")
    parser.add_argument("--data", default="./train", help="Folder dataset train/")
    parser.add_argument("--csv", default="./train_sources.csv", help="File train_sources.csv")
    parser.add_argument("--out-dir", default="./submission", help="Folder output model.pth")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4, help="LR backbone")
    parser.add_argument("--head-lr-mult", type=float, default=10.0, help="Pengali LR untuk head")
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--cutmix-prob", type=float, default=0.5)
    parser.add_argument("--crop-min", type=float, default=0.4, help="Skala minimum RandomResizedCrop")
    parser.add_argument("--erase-p", type=float, default=0.25, help="Peluang RandomErasing")
    parser.add_argument("--drop-path", type=float, default=0.1, help="Stochastic depth")
    parser.add_argument("--ema", type=float, default=0.0, help="Decay EMA bobot (0 = mati)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42, help="Seed split train/validasi")
    parser.add_argument("--init-seed", type=int, default=-1, help="Seed inisialisasi head; -1 = ikut --seed")
    parser.add_argument("--all-data", action="store_true", help="Latih di train+validasi, simpan epoch terakhir")
    parser.add_argument("--last-epoch", action="store_true", help="Simpan epoch terakhir, bukan val terbaik")
    parser.add_argument("--no-pretrained", action="store_true", help="Tanpa bobot ImageNet (uji cepat)")
    train(parser.parse_args())


if __name__ == "__main__":
    main()