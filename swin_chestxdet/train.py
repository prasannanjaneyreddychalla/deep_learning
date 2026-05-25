#!/usr/bin/env python3
# /home/pchalla7/train-29/train.py
# Hardcoded SwinV2 512x512 trainer, now robust to COCO-style detection JSON.
# No CLI flags. Reads /scratch/pchalla7/chestx-det/data/ChestX_Det_train.json
# Converts detection boxes -> per-image multi-hot labels for multi-label classification.

import os
import json
import math
import time
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch.amp import GradScaler
import timm

# =========================
# Hardcoded config
# =========================
DATA_ROOT = Path("/scratch/pchalla7/chestx-det/data")
TRAIN_DIR = DATA_ROOT / "train"
TRAIN_JSON = DATA_ROOT / "ChestX_Det_train.json"

OUT_DIR = Path("/home/pchalla7/train-29/checkpoints")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 512
IN_CHANS = 1
BATCH_SIZE = 24
EPOCHS = 20
LR = 3e-4
WEIGHT_DECAY = 1e-4
WORKERS = 8
USE_WANDB = False
WANDB_PROJECT = "chestxdet"
WANDB_RUN_NAME = "swinv2_512_hardcoded"

BACKBONE = "swinv2_tiny_window8_256"
HEAD_DIM = 768
VAL_SPLIT = 0.1
SEED = 1337
LOG_INTERVAL = 50

def set_seed(sd=SEED):
    random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)

def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# =========================
# JSON parsing helpers
# =========================

def _multi_hot(indices, K):
    v = torch.zeros(K, dtype=torch.float32)
    if isinstance(indices, (list, tuple, set)):
        for i in indices:
            ii = int(i)
            if 0 <= ii < K:
                v[ii] = 1.0
    else:
        ii = int(indices)
        if 0 <= ii < K:
            v[ii] = 1.0
    return v

def _is_coco_style(obj):
    return isinstance(obj, dict) and "images" in obj and "annotations" in obj

def parse_coco_detection(obj):
    """
    COCO-ish:
      obj = {
        "images": [{"id": 12, "file_name": "36200.png"}, ...],
        "annotations": [{"image_id": 12, "category_id": 3, "bbox": [...]}, ...],
        "categories": [{"id":3, "name":"something"}, ...]
      }
    Build per-image multi-hot from category_id across all boxes.
    """
    imgs = obj.get("images", [])
    anns = obj.get("annotations", [])
    cats = obj.get("categories", [])

    if not imgs or not anns:
        raise ValueError("COCO-style JSON must have non-empty 'images' and 'annotations'.")

    # Build id->filename
    id2name = {}
    for im in imgs:
        iid = im.get("id")
        fn = im.get("file_name") or im.get("filename") or im.get("name")
        if iid is None or fn is None:
            continue
        id2name[iid] = fn

    # Build category id mapping to [0..K-1] stable order by cat.id ascending
    if cats:
        cats_sorted = sorted(cats, key=lambda x: int(x.get("id", 0)))
        cid2idx = {int(c["id"]): i for i, c in enumerate(cats_sorted)}
        class_names = [c.get("name", f"class_{c.get('id')}") for c in cats_sorted]
        K = len(cid2idx)
    else:
        # Infer from annotations
        cids = sorted({int(a.get("category_id", 0)) for a in anns})
        cid2idx = {c:i for i,c in enumerate(cids)}
        class_names = [f"class_{c}" for c in cids]
        K = len(cid2idx)

    # Aggregate categories per image
    img_to_cidxs = defaultdict(set)
    for a in anns:
        img_id = a.get("image_id")
        cat_id = a.get("category_id")
        if img_id in id2name and cat_id is not None and int(cat_id) in cid2idx:
            img_to_cidxs[id2name[img_id]].add(cid2idx[int(cat_id)])

    # Build mapping filename -> multi-hot vector
    file_to_vec = {}
    for fn in id2name.values():
        cidxs = img_to_cidxs.get(fn, set())
        file_to_vec[fn] = _multi_hot(list(cidxs), K)

    # If literally no positives anywhere, that’s a labeling problem
    pos_sum = sum(int(v.sum().item()) for v in file_to_vec.values())
    if pos_sum == 0:
        raise ValueError("Parsed COCO JSON but found zero positive labels across images.")

    return file_to_vec, K, class_names

def parse_filename_to_boxes_mapping(obj):
    """
    Accepts formats like:
      { "36200.png": [{"label": 1, "bbox": [...]}, ...], ... }
      { "36200.png": [{"category": "Nodule"}, {"tag": "Effusion"}], ...}
    Produces per-image multi-hot with auto class indexing.
    """
    if not isinstance(obj, dict):
        raise ValueError("Expected dict mapping filename -> list of box dicts")
    class_vocab = {}
    next_id = 0

    file_to_idxset = {}
    for k, v in obj.items():
        if not isinstance(v, list):
            # maybe direct vector
            continue
        idxset = set()
        for box in v:
            if not isinstance(box, dict):
                continue
            # look for label fields
            if "label" in box:
                cat = box["label"]
            elif "category" in box:
                cat = box["category"]
            elif "tag" in box:
                cat = box["tag"]
            elif "cls" in box:
                cat = box["cls"]
            else:
                # no category in this box; ignore
                continue

            # Normalize to int ids
            if isinstance(cat, (int, float)):
                cid = int(cat)
            else:
                cname = str(cat)
                if cname not in class_vocab:
                    class_vocab[cname] = next_id
                    next_id += 1
                cid = class_vocab[cname]
            idxset.add(cid)
        file_to_idxset[k] = idxset

    if len(file_to_idxset) == 0:
        raise ValueError("Found no usable box labels in filename->boxes mapping.")

    if class_vocab:
        # string-based classes
        K = len(class_vocab)
        class_names = [""] * K
        for name, idx in class_vocab.items():
            class_names[idx] = name
    else:
        # purely numeric classes; infer K
        all_ids = set()
        for s in file_to_idxset.values():
            all_ids |= s
        K = (max(all_ids) + 1) if all_ids else 1
        class_names = [f"class_{i}" for i in range(K)]

    file_to_vec = {fn: _multi_hot(sorted(idxset), K) for fn, idxset in file_to_idxset.items()}
    pos_sum = sum(int(v.sum().item()) for v in file_to_vec.values())
    if pos_sum == 0:
        raise ValueError("Parsed mapping JSON but found zero positive labels.")
    return file_to_vec, K, class_names

def parse_generic_per_image_vectors(obj):
    """
    Fallbacks:
      - {'images': [{'file_name': 'x.png', 'labels': [0,1,0,...]}, ...]}
      - [{'path':'x.png','labels':[...]}]
      - {'x.png':[...], 'y.png':[...]}
    """
    items = []
    if isinstance(obj, dict):
        if "images" in obj and isinstance(obj["images"], list):
            for r in obj["images"]:
                fn = r.get("file_name") or r.get("path") or r.get("image") or r.get("name")
                if fn is None:
                    continue
                if "labels" in r:
                    items.append((fn, r["labels"]))
        else:
            # filename -> vector
            for k, v in obj.items():
                if isinstance(v, (list, tuple)):
                    items.append((k, v))
    elif isinstance(obj, list):
        for r in obj:
            if isinstance(r, dict):
                fn = r.get("file_name") or r.get("path") or r.get("image") or r.get("name")
                if fn is None:
                    continue
                if "labels" in r:
                    items.append((fn, r["labels"]))

    if not items:
        raise ValueError("No per-image vector labels found in generic parser.")

    K = None
    file_to_vec = {}
    for fn, vec in items:
        t = torch.tensor([float(x) for x in vec], dtype=torch.float32)
        if K is None:
            K = t.numel()
        elif t.numel() != K:
            raise ValueError("Inconsistent label vector length across images.")
        file_to_vec[fn] = t
    class_names = [f"class_{i}" for i in range(K)]
    pos_sum = sum(int(v.sum().item()) for v in file_to_vec.values())
    if pos_sum == 0:
        raise ValueError("Found vectors but all-zero across dataset.")
    return file_to_vec, K, class_names

def load_json_annotations(json_path):
    with open(json_path, "r") as f:
        obj = json.load(f)

    # Try parsers in sensible order
    parsers = []
    if _is_coco_style(obj):
        parsers.append(("COCO detection", parse_coco_detection))
    # filename->boxes mapping
    parsers.append(("filename->boxes", parse_filename_to_boxes_mapping))
    # per-image vectors
    parsers.append(("per-image vectors", parse_generic_per_image_vectors))

    last_err = None
    for name, fn in parsers:
        try:
            file_to_vec, K, class_names = fn(obj)
            print(f"[info] parsed JSON as {name}: {len(file_to_vec)} images, {K} classes")
            if class_names:
                snippet = ", ".join(class_names[:10])
                if len(class_names) > 10:
                    snippet += ", ..."
                print(f"[info] classes: {snippet}")
            return file_to_vec, K, class_names
        except Exception as e:
            last_err = e

    raise ValueError(f"Could not parse {json_path.name}. Last error: {last_err}")

# =========================
# Dataset
# =========================

class ChestXDetJSONDataset(Dataset):
    def __init__(self, root_dir: Path, file_to_vec: dict, file_list: list,
                 img_size=IMG_SIZE, in_chans=IN_CHANS, normalize=True):
        self.root_dir = Path(root_dir)
        self.file_to_vec = file_to_vec
        self.files = file_list
        self.num_classes = next(iter(file_to_vec.values())).numel()

        t = []
        if in_chans == 1:
            t.append(transforms.Grayscale(num_output_channels=1))
        t += [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ]
        if normalize:
            if in_chans == 1:
                t.append(transforms.Normalize([0.5], [0.5]))
            else:
                t.append(transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225]))
        self.tfms = transforms.Compose(t)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        p = self.root_dir / fn
        if not p.exists():
            p = Path(fn)
        img = Image.open(p).convert("RGB")
        img = self.tfms(img)
        y = self.file_to_vec[fn]
        return img, y

# =========================
# Model
# =========================

class SwinV2Classifier(nn.Module):
    def __init__(self, num_classes, img_size=IMG_SIZE, in_chans=IN_CHANS,
                 backbone_name=BACKBONE, out_indices=(1,2,3,4), pretrained=True, head_dim=HEAD_DIM):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            features_only=True,
            out_indices=out_indices,
            pretrained=pretrained,
            img_size=img_size,
            in_chans=in_chans
        )
        chs = self.backbone.feature_info.channels()
        last_c = chs[-1]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(last_c, head_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(head_dim, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        x = feats[-1]
        x = self.pool(x).flatten(1)
        logits = self.head(x)
        return logits

# =========================
# Train / Eval
# =========================

def train_one_epoch(model, loader, optimizer, device, scaler, amp_dtype=torch.float16,
                    log_interval=LOG_INTERVAL, use_amp=True, loss_fn=None):
    model.train()
    loss_fn = loss_fn or nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n = 0
    t0 = time.time()

    for step, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                logits = model(imgs)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)

        if (step + 1) % log_interval == 0:
            print(f"[train] step {step+1}/{len(loader)} | loss {total_loss / n:.4f}")

    return total_loss / max(1, n), time.time() - t0

@torch.no_grad()
def evaluate(model, loader, device, loss_fn=None, amp_dtype=torch.float16, use_amp=True):
    model.eval()
    loss_fn = loss_fn or nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                logits = model(imgs)
                loss = loss_fn(logits, labels)
        else:
            logits = model(imgs)
            loss = loss_fn(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / max(1, n)

def init_wandb(model, train_count, val_count, num_params_m):
    if not USE_WANDB:
        return None
    try:
        import wandb
        wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, config={
            "img_size": IMG_SIZE,
            "in_chans": IN_CHANS,
            "backbone": BACKBONE,
            "head_dim": HEAD_DIM,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
        })
        wandb.watch(model, log="all", log_freq=200)
        wandb.log({"train_samples": train_count, "val_samples": val_count, "params_million": num_params_m})
        return wandb
    except Exception as e:
        print(f"[warn] wandb not initialized: {e}")
        return None

# =========================
# Main
# =========================

def main():
    set_seed()

    # Parse JSON into per-image label vectors
    file_to_vec, num_classes, class_names = load_json_annotations(TRAIN_JSON)

    # Keep only files that actually exist in train dir
    exist_files = []
    skipped = 0
    for fn in file_to_vec.keys():
        p = TRAIN_DIR / fn
        if p.exists() or Path(fn).exists():
            exist_files.append(fn)
        else:
            skipped += 1
    if skipped:
        print(f"[warn] {skipped} JSON entries missing under {TRAIN_DIR}. Skipped.")

    if not exist_files:
        raise RuntimeError("No images found that match JSON filenames in train/.")

    # Split train/val
    random.shuffle(exist_files)
    val_n = max(1, int(len(exist_files) * VAL_SPLIT))
    val_files = exist_files[:val_n]
    train_files = exist_files[val_n:]

    ds_train = ChestXDetJSONDataset(TRAIN_DIR, file_to_vec, train_files, img_size=IMG_SIZE, in_chans=IN_CHANS, normalize=True)
    ds_val   = ChestXDetJSONDataset(TRAIN_DIR, file_to_vec, val_files,   img_size=IMG_SIZE, in_chans=IN_CHANS, normalize=True)

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=WORKERS, pin_memory=True, drop_last=False)
    val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=WORKERS, pin_memory=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    model = SwinV2Classifier(
        num_classes=num_classes,
        img_size=IMG_SIZE,
        in_chans=IN_CHANS,
        backbone_name=BACKBONE,
        out_indices=(1,2,3,4),
        pretrained=True,
        head_dim=HEAD_DIM
    ).to(device)

    print(f"[info] params: {count_parameters(model)/1e6:.2f}M | device: {device} | img_size: {IMG_SIZE} | in_chans: {IN_CHANS} | classes: {num_classes}")
    if class_names:
        print("[info] class list head:", ", ".join(class_names[:10]) + ("..." if len(class_names) > 10 else ""))
    try:
        print("patch_embed img_size =", model.backbone.patch_embed.img_size)
    except Exception:
        pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler('cuda', enabled=use_amp)
    wb = init_wandb(model, len(ds_train), len(ds_val), count_parameters(model)/1e6)

    best_val = math.inf
    best_path = None

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_time = train_one_epoch(model, train_loader, optimizer, device, scaler, use_amp=use_amp)
        va_loss = evaluate(model, val_loader, device, use_amp=use_amp)

        print(f"[epoch {epoch:03d}] train_loss={tr_loss:.4f} ({tr_time:.1f}s) | val_loss={va_loss:.4f}")
        if wb:
            try:
                wb.log({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "lr": optimizer.param_groups[0]["lr"]})
            except Exception:
                pass

        if va_loss < best_val:
            best_val = va_loss
            best_path = OUT_DIR / f"model_best_epoch{epoch:03d}_valloss{va_loss:.4f}.pt"
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_loss": va_loss,
                "img_size": IMG_SIZE,
                "in_chans": IN_CHANS,
                "backbone": BACKBONE,
                "head_dim": HEAD_DIM,
                "classes": num_classes,
                "class_names": class_names
            }, best_path)
            print(f"[info] saved best: {best_path}")

    print(f"[done] best_val_loss={best_val:.4f} | best_path={best_path}")

if __name__ == "__main__":
    main()

