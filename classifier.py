import argparse
import json
import os

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from face_utils import CROP_SIZE, MEAN, STD

GENUINE_CLASS_NAMES = {"live", "real"}


def build_loaders(data_dir, batch_size, workers):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(CROP_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((CROP_SIZE, CROP_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), val_tf)
    if train_ds.classes != val_ds.classes:
        raise SystemExit(f"train classes {train_ds.classes} != val classes {val_ds.classes}")
    return train_ds, val_ds, (
        DataLoader(train_ds, batch_size, shuffle=True, num_workers=workers, pin_memory=True),
        DataLoader(val_ds, batch_size, shuffle=False, num_workers=workers, pin_memory=True),
    )


@torch.no_grad()
def evaluate(model, loader, device, genuine_idx):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device)), dim=1)[:, genuine_idx]
        probs.append(p.cpu().numpy())
        labels.append(y.numpy())
    probs, labels = np.concatenate(probs), np.concatenate(labels)

    is_genuine = labels == genuine_idx
    pred_genuine = probs >= 0.5
    apcer = float(np.mean(pred_genuine[~is_genuine])) if (~is_genuine).any() else float("nan")
    bpcer = float(np.mean(~pred_genuine[is_genuine])) if is_genuine.any() else float("nan")
    try:
        auc = float(roc_auc_score(is_genuine, probs))
    except ValueError:
        auc = float("nan")
    return {
        "acc": float(np.mean(pred_genuine == is_genuine)),
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": (apcer + bpcer) / 2,
        "auc": auc,
    }


def export_onnx(model, out_path, classes):
    model = model.cpu().eval()
    dummy = torch.randn(1, 3, CROP_SIZE, CROP_SIZE)
    kwargs = dict(input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(model, dummy, out_path, dynamo=False, **kwargs)
    except TypeError:  # older PyTorch without the `dynamo` argument
        torch.onnx.export(model, dummy, out_path, **kwargs)
    with open(os.path.splitext(out_path)[0] + ".classes.json", "w") as f:
        json.dump(classes, f)
    print(f"Exported {out_path} (+ classes.json)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="folder containing train/ and val/")
    p.add_argument("--out", default="model.onnx")
    p.add_argument("--model", default="mobilenetv3_large_100", help="any timm model name")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no-pretrained", action="store_true", help="train from scratch (for quick tests)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, val_ds, (train_dl, val_dl) = build_loaders(args.data, args.batch_size, args.workers)
    classes = train_ds.classes
    genuine = [i for i, c in enumerate(classes) if c.lower() in GENUINE_CLASS_NAMES]
    if len(classes) != 2 or not genuine:
        raise SystemExit(f"Need exactly 2 classes incl. 'live' or 'real'; found {classes}")
    genuine_idx = genuine[0]
    print(f"Classes: {classes} (genuine = {classes[genuine_idx]}) | "
          f"train {len(train_ds)} / val {len(val_ds)} | device {device}")

    # Weight the loss so an imbalanced dataset doesn't just predict the majority class
    counts = np.bincount(train_ds.targets, minlength=2).astype(np.float32)
    weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), device=device)

    model = timm.create_model(args.model, pretrained=not args.no_pretrained, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_acer, best_path = float("inf"), os.path.splitext(args.out)[0] + "_best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
        scheduler.step()

        m = evaluate(model, val_dl, device, genuine_idx)
        print(f"epoch {epoch}: loss {running / len(train_ds):.4f} | acc {m['acc']:.3f} | "
              f"APCER {m['apcer']:.3f} | BPCER {m['bpcer']:.3f} | ACER {m['acer']:.3f} | AUC {m['auc']:.3f}")
        if m["acer"] < best_acer:
            best_acer = m["acer"]
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location="cpu"))
    print(f"Best val ACER: {best_acer:.3f}")
    export_onnx(model, args.out, classes)


if __name__ == "__main__":
    main()