#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import json
import csv
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.models as tvm

from sklearn.metrics import roc_auc_score, roc_curve

# =========================
# HARD-CODED CONFIG (edit here)
# =========================
CONFIG: Dict[str, Any] = {
    # Split-specific roots
    "TRAIN_ROOT": "/scratch/pchalla7/full_uncompressed",
    "VALID_ROOT": "/scratch/pchalla7/full_uncompressed",   # valid is under CheXpert root
    "TEST_ROOT":  "/scratch/pchalla7/chexpert/chexlocalize/CheXpert",

    # CSVs
    "TRAIN_CSV": "/scratch/pchalla7/full_uncompressed/train.csv",
    "VALID_CSV": "/scratch/pchalla7/full_uncompressed/valid.csv",
    "TEST_CSV":  "/scratch/pchalla7/chexpert/chexlocalize/CheXpert/test_labels.csv",

    # Preflight audit first; flip to False to train
    "PRECHECK_ONLY": False,

    # Auto-drop rows whose image files are missing (filtered CSVs saved to ./outputs)
    "AUTO_FILTER_MISSING": True,

    # Training
    "EPOCHS": 100,
    "BATCH_SIZE": 200,                 # increase on full A100; target ~70–85% VRAM usage
    "LR": 1e-4,
    "WEIGHT_DECAY": 1e-4,
    "NUM_WORKERS": 8,                 # try 8–12 if CPU available
    "SEED": 42,
    "IMAGE_SIZE": 224,
    "UNCERTAINTY_POLICY": "u-one",    # "u-zero" | "u-one" | "ignore"
    "PATIENCE": 5,
    "FP16": True,                     # stays True; uses new torch.amp API

    # Weights & Biases (kept offline; dirs unchanged)
    "WANDB_PROJECT": "chexpert-convnext",
    "WANDB_RUN_NAME_PREFIX": "convnext_base_u-one",

    # Output (unchanged)
    "OUTPUT_DIR": "./outputs",

    # ImageNet-1K init
    "USE_PRETRAINED": True,

    # Logging cadence
    "LOG_EVERY_STEPS": 50,            # prints throughput during training
}

# Optional per-epoch emails. Set before running:
# export MAIL_TO="you@asu.edu"
MAIL_TO = os.environ.get("MAIL_TO", "").strip()

# =========================
# Constants
# =========================
CHEXPERT_14 = [
    "No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion",
    "Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax",
    "Pleural Effusion","Pleural Other","Fracture","Support Devices",
]
MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET  = [0.229, 0.224, 0.225]

# W&B optional, forced offline later
try:
    import wandb
    WANDB_LIB_AVAILABLE = True
except Exception:
    WANDB_LIB_AVAILABLE = False

# =========================
# Utilities
# =========================
def seed_everything(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def notify(subject: str, message: str):
    if not MAIL_TO:
        return
    try:
        proc = subprocess.run(["bash", "-lc", f'echo "{message}" | mail -s "{subject}" "{MAIL_TO}"'],
                              check=False, capture_output=True, text=True)
        if proc.returncode == 0:
            return
    except Exception:
        pass
    try:
        p = subprocess.Popen(["/usr/sbin/sendmail", MAIL_TO], stdin=subprocess.PIPE)
        p.communicate(f"Subject: {subject}\n\n{message}\n".encode("utf-8"))
    except Exception:
        pass

def resolve_path(base_dir: Path, path_in_csv: str) -> Path:
    p = Path(path_in_csv)
    if p.is_absolute() and p.exists():
        return p
    cand = base_dir / path_in_csv
    if cand.exists():
        return cand
    for i in range(len(p.parts)):
        maybe = base_dir.joinpath(*p.parts[i:])
        if maybe.exists():
            return maybe
    return cand

def _detect_path_column(df: pd.DataFrame) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    candidates = ["path", "image", "study", "studypath", "study_path", "studyid", "study_id"]
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    first = df.columns[0]
    sample_vals = df[first].astype(str).head(10).tolist()
    if any(re.search(r"(patient|study|chexpert|test|train|valid|/|\.jpg|\.png|\.jpeg)", s, re.I) for s in sample_vals):
        return first
    raise ValueError(f"Could not locate a path/study column. Columns found: {list(df.columns)}")

def load_csv_labels(csv_path: Path, classes: List[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    path_col = _detect_path_column(df)

    colmap_lower = {c.lower(): c for c in df.columns}
    have = {}
    missing = []
    for cls in classes:
        key = cls.lower()
        if key in colmap_lower:
            have[cls] = colmap_lower[key]
        else:
            alt = key.replace(" ", "_")
            found = None
            for lc, orig in colmap_lower.items():
                if lc == alt or lc.replace("_"," ") == key:
                    found = orig; break
            if found: have[cls] = found
            else:     missing.append(cls)

    cols = [path_col] + [have[c] for c in have]
    out = df[cols].copy()
    out.rename(columns={path_col: "Path", **{v:k for k,v in have.items()}}, inplace=True)
    for cls in missing:
        out[cls] = np.nan
    if missing:
        print(f"[warn] {csv_path} missing label columns, filled with NaN: {missing}")
    out = out[["Path"] + classes]
    return out

def apply_uncertainty_policy(labels: np.ndarray, policy: str) -> np.ndarray:
    arr = labels.copy()
    if policy == "u-zero": arr[arr == -1] = 0
    elif policy == "u-one": arr[arr == -1] = 1
    elif policy == "ignore": pass
    else: raise ValueError("uncertainty_policy must be u-zero, u-one, or ignore")
    return arr

def compute_pos_weight(y: np.ndarray, policy: str) -> torch.Tensor:
    y2 = y.copy()
    y2[np.isnan(y2)] = -1
    y2 = apply_uncertainty_policy(y2, policy)
    weights = []
    for c in range(y2.shape[1]):
        col = y2[:, c]
        if policy == "ignore":
            col = col[col >= 0]
        pos = np.sum(col == 1)
        neg = np.sum(col == 0)
        weights.append(1.0 if pos == 0 else max(1.0, float(neg)/float(pos)))
    return torch.tensor(weights, dtype=torch.float32)

def auc_per_class(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[Dict[str, float], float]:
    out = {}
    for i, name in enumerate(CHEXPERT_14):
        true, prob = y_true[:, i], y_prob[:, i]
        mask = ~np.isnan(true)
        try:
            out[name] = roc_auc_score(true[mask], prob[mask]) if mask.sum() > 1 and len(np.unique(true[mask])) > 1 else float("nan")
        except Exception:
            out[name] = float("nan")
    vals = [v for v in out.values() if v == v]
    return out, (float(np.mean(vals)) if vals else float("nan"))

def roc_curves_per_class(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    curves: Dict[str, Dict[str, np.ndarray]] = {}
    for i, name in enumerate(CHEXPERT_14):
        true, prob = y_true[:, i], y_prob[:, i]
        mask = ~np.isnan(true)
        if mask.sum() > 1 and len(np.unique(true[mask])) > 1:
            fpr, tpr, thr = roc_curve(true[mask], prob[mask])
            curves[name] = {"fpr": fpr, "tpr": tpr, "thr": thr}
    return curves

def normalize_abs(base_dir: Path, p: str) -> str:
    abs_p = resolve_path(base_dir, p).resolve()
    return str(abs_p).replace("\\", "/").lower()

def audit_dataset_split(base_dir: Path, df: pd.DataFrame, name: str) -> Dict[str, int]:
    rels = df["Path"].astype(str).tolist()
    exists = sum((resolve_path(base_dir, r).exists() for r in rels))
    rel_norm = {normalize_abs(base_dir, r) for r in rels}
    return {
        f"{name}_count": len(rels),
        f"{name}_exists": exists,
        f"{name}_missing": len(rels) - exists,
        f"{name}_unique_paths": len(rel_norm),
    }

def audit_datasets(train_root: Path, df_train: pd.DataFrame,
                   valid_root: Path, df_valid: pd.DataFrame,
                   test_root: Optional[Path], df_test: Optional[pd.DataFrame]) -> Dict[str, int]:
    report: Dict[str, int] = {}
    report.update(audit_dataset_split(train_root, df_train, "train"))
    report.update(audit_dataset_split(valid_root, df_valid, "valid"))
    if df_test is not None and test_root is not None:
        report.update(audit_dataset_split(test_root, df_test, "test"))

    train_set = {normalize_abs(train_root, p) for p in df_train["Path"].astype(str)}
    valid_set = {normalize_abs(valid_root, p) for p in df_valid["Path"].astype(str)}
    report["overlap_train_valid"] = len(train_set & valid_set)
    if df_test is not None and test_root is not None:
        test_set = {normalize_abs(test_root, p) for p in df_test["Path"].astype(str)}
        report["overlap_train_test"] = len(train_set & test_set)
        report["overlap_valid_test"] = len(valid_set & test_set)
    return report

def label_sanity(df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    if df is None:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    arr = df[CHEXPERT_14].to_numpy(dtype=np.float32)
    for i, cls in enumerate(CHEXPERT_14):
        col = arr[:, i]
        nan_rate = float(np.isnan(col).mean())
        pos_rate = float(np.nanmean((col == 1).astype(np.float32))) if np.sum(~np.isnan(col)) > 0 else float("nan")
        out[cls] = {"nan_rate": nan_rate, "pos_rate": pos_rate}
    return out

def filter_missing(df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    keep = []
    for p in df["Path"].astype(str).tolist():
        keep.append(resolve_path(base_dir, p).exists())
    return df.loc[keep].reset_index(drop=True)

def save_metrics_row(csv_path: Path, row: Dict[str, Any]):
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def save_curves_npz(npz_path: Path, curves: Dict[str, Dict[str, np.ndarray]]):
    arrays = {}
    for name, d in curves.items():
        arrays[f"{name}_fpr"] = d["fpr"]
        arrays[f"{name}_tpr"] = d["tpr"]
        arrays[f"{name}_thr"] = d["thr"]
    np.savez_compressed(npz_path, **arrays)

def save_preds_csv(csv_path: Path, paths: List[str], y_true: np.ndarray, y_prob: np.ndarray):
    cols = ["Path"] + [f"true_{c}" for c in CHEXPERT_14] + [f"prob_{c}" for c in CHEXPERT_14]
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, p in enumerate(paths):
            row = [p] + list(y_true[i]) + list(y_prob[i])
            w.writerow(row)

# =========================
# Dataset
# =========================
class CheXpertDataset(Dataset):
    def __init__(self, base_dir: Path, csv_path: Path, classes: List[str], split: str,
                 uncertainty_policy: str = "u-one", transform=None):
        self.base_dir = Path(base_dir)
        self.df = load_csv_labels(csv_path, classes)
        self.classes = classes
        self.split = split
        self.policy = uncertainty_policy
        self.transform = transform
        self.labels_raw = self.df[self.classes].to_numpy(dtype=np.float32)
        self.labels_for_loss = apply_uncertainty_policy(self.labels_raw.copy(), self.policy)
        self.paths = self.df["Path"].tolist()
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        rel = self.paths[idx]
        img_path = resolve_path(self.base_dir, rel)
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            if self.transform: im = self.transform(im)
        y_raw = self.labels_raw[idx].copy()
        y_loss = self.labels_for_loss[idx].copy()
        ignore_mask = (y_loss == -1).astype(np.float32)
        y_loss = np.nan_to_num(y_loss, nan=0.0)
        return (im, torch.from_numpy(y_loss).float(),
                torch.from_numpy(ignore_mask).float(),
                torch.from_numpy(y_raw).float(),
                rel)

# =========================
# Model
# =========================
def build_convnext_base(num_classes: int, pretrained: bool = True):
    weights = tvm.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.convnext_base(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model

# =========================
# Train/Eval
# =========================
def train_one_epoch(model, loader, optimizer, scaler, device, pos_weight, ignore_policy):
    model.train()
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight.to(device))
    total, count = 0.0, 0

    # throughput logging
    LOG_EVERY = int(CONFIG.get("LOG_EVERY_STEPS", 50))
    t0 = time.time()
    seen = 0

    for step, (x, y, ignore_mask, _, _) in enumerate(loader, 1):
        # to device + channels-last
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        ignore_mask = ignore_mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=CONFIG["FP16"]):
            logits = model(x)
            loss_all = bce(logits, y)
            if ignore_policy == "ignore":
                loss_all = loss_all * (1.0 - ignore_mask)
            loss = loss_all.mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += loss.item() * x.size(0)
        count += x.size(0)
        seen += x.size(0)

        if WANDB_ACTIVE:
            wandb.log({"train/loss_step": loss.item()})

        if LOG_EVERY > 0 and step % LOG_EVERY == 0:
            dt = time.time() - t0
            ips = seen / max(dt, 1e-6)
            mem_gb = torch.cuda.memory_allocated() / 1e9
            print(f"[train step {step}] {ips:.1f} img/s | VRAM={mem_gb:.2f} GB")

    return total / max(1, count)

@torch.no_grad()
def evaluate(model, loader, device, pos_weight, ignore_policy, compute_curves: bool = False):
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight.to(device))
    loss_sum, nsum = 0.0, 0
    probs_all, ytrue_all, paths_all = [], [], []

    for x, y_loss, ignore_mask, y_raw, rel in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        y_loss = y_loss.to(device, non_blocking=True)
        ignore_mask = ignore_mask.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=CONFIG["FP16"]):
            logits = model(x)
            probs = torch.sigmoid(logits)
            loss_all = bce(logits, y_loss)
            if ignore_policy == "ignore":
                loss_all = loss_all * (1.0 - ignore_mask)
            loss = loss_all.mean()

        loss_sum += loss.item() * x.size(0)
        nsum += x.size(0)
        probs_all.append(probs.cpu())
        ytrue_all.append(y_raw.cpu())
        paths_all.extend(rel)

    probs = torch.cat(probs_all, 0).numpy()
    y_true = torch.cat(ytrue_all, 0).numpy()
    per_cls_auc, mean_auc = auc_per_class(y_true, probs)
    curves = roc_curves_per_class(y_true, probs) if compute_curves else {}
    return {
        "loss": loss_sum / max(1, nsum),
        "per_class_auc": per_cls_auc,
        "mean_auc": mean_auc,
        "probs": probs,
        "y_true": y_true,
        "paths": paths_all,
        "curves": curves
    }

# =========================
# Main
# =========================
def main():
    seed_everything(CONFIG["SEED"])

    # Enable TF32 on Ampere and cuDNN autotune
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(CONFIG["OUTPUT_DIR"]); outdir.mkdir(parents=True, exist_ok=True)

    # Force W&B offline local logging (keeps your wandb dir behavior unchanged)
    os.environ["WANDB_MODE"] = "offline"
    os.environ.setdefault("WANDB_PROJECT", CONFIG["WANDB_PROJECT"])

    train_root = Path(CONFIG["TRAIN_ROOT"])
    valid_root = Path(CONFIG["VALID_ROOT"])
    test_root  = Path(CONFIG["TEST_ROOT"])

    df_train = load_csv_labels(Path(CONFIG["TRAIN_CSV"]), CHEXPERT_14)
    df_valid = load_csv_labels(Path(CONFIG["VALID_CSV"]), CHEXPERT_14)
    test_csv_path = Path(CONFIG["TEST_CSV"])
    df_test = load_csv_labels(test_csv_path, CHEXPERT_14) if test_csv_path.exists() else None
    if df_test is None:
        print(f"[warn] TEST_CSV not found at {test_csv_path}. Test evaluation will be skipped.")

    # Preflight audit across roots
    audit = audit_datasets(train_root, df_train, valid_root, df_valid, test_root, df_test)
    sanity_train, sanity_valid = label_sanity(df_train), label_sanity(df_valid)
    sanity_test = label_sanity(df_test) if df_test is not None else {}

    print("\n=== DATA AUDIT SUMMARY ===")
    for k, v in audit.items():
        print(f"{k}: {v}")

    def summarize_rates(name, stats):
        if not stats:
            print(f"{name}: N/A"); return
        nan_mean = float(np.mean([d['nan_rate'] for d in stats.values()]))
        pos_vals = [d['pos_rate'] for d in stats.values() if d['pos_rate'] == d['pos_rate']]
        pos_mean = float(np.mean(pos_vals)) if pos_vals else float("nan")
        print(f"{name}: mean_nan_rate={nan_mean:.3f}, mean_pos_rate={pos_mean:.3f}")

    print("\n=== LABEL SANITY (rates) ===")
    summarize_rates("train", sanity_train)
    summarize_rates("valid", sanity_valid)
    summarize_rates("test", sanity_test)

    problems = []
    if audit.get("overlap_train_valid", 0) > 0:
        problems.append(f"Train/Valid overlap: {audit['overlap_train_valid']}")
    if "overlap_train_test" in audit and audit.get("overlap_train_test", 0) > 0:
        problems.append(f"Train/Test overlap: {audit['overlap_train_test']}")
    if "overlap_valid_test" in audit and audit.get("overlap_valid_test", 0) > 0:
        problems.append(f"Valid/Test overlap: {audit['overlap_valid_test']}")
    if df_test is not None and audit.get("test_missing", 0) > 0:
        problems.append(f"Test images missing: {audit['test_missing']} under TEST_ROOT.")

    # Optional: write filtered CSVs that only contain rows with files that exist
    filtered_train_csv = Path(CONFIG["OUTPUT_DIR"]) / "filtered_train.csv"
    filtered_valid_csv = Path(CONFIG["OUTPUT_DIR"]) / "filtered_valid.csv"

    if CONFIG.get("AUTO_FILTER_MISSING", False):
        before_tr, before_va = len(df_train), len(df_valid)
        df_train_f = filter_missing(df_train, train_root)
        df_valid_f = filter_missing(df_valid, valid_root)
        df_train_f.to_csv(filtered_train_csv, index=False)
        df_valid_f.to_csv(filtered_valid_csv, index=False)
        print(f"\n[filter] train: {before_tr} -> {len(df_train_f)} rows after dropping missing files")
        print(f"[filter] valid: {before_va} -> {len(df_valid_f)} rows after dropping missing files")

    notify("CheXpert precheck complete", "Audit done.\n" + "\n".join(f"{k}={v}" for k, v in audit.items()))

    if CONFIG["PRECHECK_ONLY"]:
        if problems:
            print("\n[FAIL] Fix these before training (or accept and proceed):")
            for p in problems: print(" -", p)
            if CONFIG.get("AUTO_FILTER_MISSING", False):
                print(" - Missing train/valid rows will be auto-filtered using CSVs in ./outputs/")
        else:
            print("\n[OK] No split overlap detected. Paths look sane.")
        print("\nSet CONFIG['PRECHECK_ONLY'] = False to start training.")
        return

    # Choose CSVs for the actual run
    train_csv_for_run = Path(CONFIG["TRAIN_CSV"])
    valid_csv_for_run = Path(CONFIG["VALID_CSV"])
    if CONFIG.get("AUTO_FILTER_MISSING", False):
        if filtered_train_csv.exists():
            train_csv_for_run = filtered_train_csv
        if filtered_valid_csv.exists():
            valid_csv_for_run = filtered_valid_csv

    # W&B init for training (forced offline)
    global WANDB_ACTIVE
    WANDB_ACTIVE = False
    run_name = f'{CONFIG["WANDB_RUN_NAME_PREFIX"]}_{int(time.time())}'
    if WANDB_LIB_AVAILABLE:
        try:
            wandb.init(
                project=os.getenv("WANDB_PROJECT", CONFIG["WANDB_PROJECT"]),
                entity=os.getenv("WANDB_ENTITY", None),
                name=run_name,
                config=CONFIG,
                mode="offline",
            )
            wandb.config.update({"classes": CHEXPERT_14}, allow_val_change=True)
            WANDB_ACTIVE = True
        except Exception as e:
            print(f"[warn] wandb.init failed: {e}\n[warn] continuing without W&B logging.")
    else:
        print("[warn] wandb not installed; continuing without W&B.")

    # Transforms
    train_tf = T.Compose([
        T.Resize((CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"])),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(MEAN_IMAGENET, STD_IMAGENET),
    ])
    eval_tf = T.Compose([
        T.Resize((CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"])),
        T.ToTensor(),
        T.Normalize(MEAN_IMAGENET, STD_IMAGENET),
    ])

    # Datasets
    train_ds = CheXpertDataset(train_root, train_csv_for_run, CHEXPERT_14, "train",
                               uncertainty_policy=CONFIG["UNCERTAINTY_POLICY"], transform=train_tf)
    valid_ds = CheXpertDataset(valid_root, valid_csv_for_run, CHEXPERT_14, "valid",
                               uncertainty_policy=CONFIG["UNCERTAINTY_POLICY"], transform=eval_tf)
    test_ds = CheXpertDataset(test_root, test_csv_path, CHEXPERT_14, "test",
                               uncertainty_policy=CONFIG["UNCERTAINTY_POLICY"], transform=eval_tf) if df_test is not None else None

    # DataLoaders (improved prefetching)
    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=True,
        num_workers=CONFIG["NUM_WORKERS"], pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
        num_workers=CONFIG["NUM_WORKERS"], pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
        num_workers=CONFIG["NUM_WORKERS"], pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    ) if test_ds else None

    # Model/optim
    pos_weight = compute_pos_weight(train_ds.labels_raw, CONFIG["UNCERTAINTY_POLICY"])
    model = build_convnext_base(len(CHEXPERT_14), pretrained=CONFIG["USE_PRETRAINED"])
    # channels-last friendly
    model = model.to(memory_format=torch.channels_last).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LR"], weight_decay=CONFIG["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["EPOCHS"])
    scaler = torch.amp.GradScaler('cuda', enabled=CONFIG["FP16"])

    best_val, best_auc = float("inf"), -1.0
    best_path = Path(CONFIG["OUTPUT_DIR"]) / "best.pt"
    metrics_csv = Path(CONFIG["OUTPUT_DIR"]) / "metrics_log.csv"

    notify("CheXpert run started", f"Training on {os.uname().nodename}. Offline W&B logging is enabled.")

    for epoch in range(1, CONFIG["EPOCHS"] + 1):
        t_epoch = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, pos_weight, CONFIG["UNCERTAINTY_POLICY"])
        # Compute curves on validation each epoch
        val = evaluate(model, valid_loader, device, pos_weight, CONFIG["UNCERTAINTY_POLICY"], compute_curves=True)
        scheduler.step()

        # Save ROC curves and predictions for validation
        roc_npz = Path(CONFIG["OUTPUT_DIR"]) / f"roc_val_epoch{epoch}.npz"
        save_curves_npz(roc_npz, val["curves"])
        val_pred_csv = Path(CONFIG["OUTPUT_DIR"]) / f"val_predictions_epoch{epoch}.csv"
        save_preds_csv(val_pred_csv, val["paths"], val["y_true"], val["probs"])

        # Log to W&B (offline)
        if WANDB_ACTIVE:
            payload = {"epoch": epoch, "train/loss": train_loss, "val/loss": val["loss"], "val/auc_mean": val["mean_auc"],
                       "lr": scheduler.get_last_lr()[0]}
            for k, v in val["per_class_auc"].items():
                payload[f"val/auc_{k}"] = v
            wandb.log(payload)

        # Append to metrics CSV
        row = {"epoch": epoch, "train_loss": f"{train_loss:.6f}", "val_loss": f"{val['loss']:.6f}", "val_auc_mean": f"{val['mean_auc']:.6f}",
               "lr": f"{scheduler.get_last_lr()[0]:.8f}"}
        for k, v in val["per_class_auc"].items():
            row[f"val_auc_{k}"] = f"{(v if v == v else float('nan')):.6f}" if v == v else ""
        save_metrics_row(metrics_csv, row)

        # Print and email summary
        cls_pairs = ", ".join(
            f"{k}:{(val['per_class_auc'][k] if val['per_class_auc'][k]==val['per_class_auc'][k] else float('nan')):.3f}"
            for k in ["Edema", "Pleural Effusion", "Atelectasis", "Cardiomegaly"]
        )
        msg = (f"Epoch {epoch}/{CONFIG['EPOCHS']}  "
               f"train_loss={train_loss:.4f}  val_loss={val['loss']:.4f}  val_auc_mean={val['mean_auc']:.4f}  "
               f"time={time.time()-t_epoch:.1f}s\n"
               f"AUC highlights: {cls_pairs}\n"
               f"Saved: {roc_npz.name}, {val_pred_csv.name}, metrics_log.csv")
        print(msg)
        notify(f"CheXpert epoch {epoch} complete", msg)

        # Early stop logic
        improved = val["loss"] < best_val - 1e-5
        if improved:
            best_val = val["loss"]
            best_auc = max(best_auc, val["mean_auc"] if val["mean_auc"] == val["mean_auc"] else -1.0)
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_auc": val["mean_auc"]}, best_path)

        # patience check
        # simplistic: stop if no improvement for PATIENCE epochs
        # track via a small list of recent bests
        # Here, we implement minimal: break if epoch - last_best_epoch >= patience
        # We store last_best_epoch in a closure variable
        if not hasattr(main, "_last_best_epoch"):
            main._last_best_epoch = epoch if improved else 0
        if improved:
            main._last_best_epoch = epoch
        if epoch - main._last_best_epoch >= CONFIG["PATIENCE"]:
            print(f"[early-stop] no improvement for {CONFIG['PATIENCE']} epochs")
            break

    # Load best for final eval
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])

    # Final test evaluation (with curves + predictions) if test exists
    if test_loader is not None:
        test = evaluate(model, test_loader, device, pos_weight, CONFIG["UNCERTAINTY_POLICY"], compute_curves=True)
        test_msg = f"[test] loss={test['loss']:.4f}  mean AUC={test['mean_auc']:.4f}"
        print(test_msg)
        notify("CheXpert test complete", test_msg)

        # Save test curves and predictions
        save_curves_npz(Path(CONFIG["OUTPUT_DIR"]) / "roc_test_final.npz", test["curves"])
        save_preds_csv(Path(CONFIG["OUTPUT_DIR"]) / "test_predictions_final.csv", test["paths"], test["y_true"], test["probs"])

        # Log to W&B offline
        if WANDB_ACTIVE:
            payload = {"test/auc_mean": test["mean_auc"], "test/loss_proxy": test["loss"]}
            for k, v in test["per_class_auc"].items():
                payload[f"test/auc_{k}"] = v
            wandb.log(payload)

        # Append to metrics CSV
        row = {"epoch": "final_test", "train_loss": "", "val_loss": "", "val_auc_mean": ""}
        row.update({f"test_auc_{k}": f"{(v if v == v else float('nan')):.6f}" for k, v in test["per_class_auc"].items()})
        row["test_auc_mean"] = f"{test['mean_auc']:.6f}"
        row["test_loss"] = f"{test['loss']:.6f}"
        save_metrics_row(metrics_csv, row)
    else:
        print("[info] Test evaluation skipped (no test dataset).")

    # Save final model
    final_path = Path(CONFIG["OUTPUT_DIR"]) / "final.pt"
    torch.save({"model": model.state_dict(), "best_val_loss": best_val, "best_val_auc": best_auc}, final_path)

    # Close W&B run (still offline)
    if WANDB_ACTIVE:
        if best_path.exists(): wandb.save(best_path.as_posix())
        wandb.save(final_path.as_posix())
        wandb.save(metrics_csv.as_posix())
        wandb.finish()

    notify("CheXpert run finished",
           f"Finished. best_val_loss={best_val:.4f}, best_val_auc={best_auc:.4f}\n"
           f"Saved model: {final_path}\nSaved metrics: {metrics_csv}\n"
           f"ROC (val per-epoch + test final) saved under {Path(CONFIG['OUTPUT_DIR']).resolve()}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        notify("CheXpert run failed", f"Failure: {e}")
        raise

