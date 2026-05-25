import os
import torch
import wandb
import pandas as pd
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from medpy.io import load
import csv

# ==========================
# CONFIGURATION (HARDCODED)
# ==========================
DATA_ROOT = "/scratch/pchalla7/node21/data/cxr_images/proccessed_data"
OUTPUT_DIR = "/home/pchalla7/outputs"
TRAIN_CSV = "/scratch/pchalla7/node21/train_boxes.csv"
VAL_CSV = "/scratch/pchalla7/node21/val_boxes.csv"
DRY_RUN = False
BATCH_SIZE = 8
IMAGE_SIZE = 224
EPOCHS = 20

os.environ["WANDB_MODE"] = "offline"  # Store wandb logs locally

# ==========================
# IMAGE RESIZE/PADDING UTILS
# ==========================
class ResizeWithPadding(object):
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        old_size = img.size
        ratio = float(self.target_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = img.resize(new_size, Image.BILINEAR)
        new_img = Image.new("RGB", (self.target_size, self.target_size))
        paste_pos = ((self.target_size - new_size[0]) // 2,
                     (self.target_size - new_size[1]) // 2)
        new_img.paste(img, paste_pos)
        return new_img

# ==========================
# DATASET (robust with missing files)
# ==========================
class Node21LocalizationDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Robust filename support
        if 'img_name' in row:
            img_name = row['img_name']
        elif 'filename' in row:
            img_name = row['filename']
        elif 'image_path' in row:
            img_name = row['image_path']
        else:
            raise ValueError(f"No valid image filename column in row:\n{row}")

        # Try main directory, then images/, then original_data/images/
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.isfile(img_path):
            alt_path = os.path.join(self.img_dir, "images", img_name)
            if os.path.isfile(alt_path):
                img_path = alt_path
            else:
                alt2 = os.path.join(os.path.dirname(self.img_dir), "original_data", "images", img_name)
                if os.path.isfile(alt2):
                    img_path = alt2
                else:
                    print(f"Warning: Image file not found for {img_name}, skipping this sample.")
                    dummy_img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
                    bbox_tensor = torch.tensor([0,0,1,1], dtype=torch.float32)
                    if self.transform:
                        dummy_img = self.transform(dummy_img)
                    return dummy_img, bbox_tensor

        # MedPy for .mha, PIL for other formats
        if img_name.endswith('.mha'):
            img_data, _ = load(img_path)
            img2d = img_data.squeeze()
            norm_img = (img2d / np.max(img2d) * 255).astype(np.uint8)
            img = Image.fromarray(norm_img)
            img = img.convert("RGB")
        else:
            img = Image.open(img_path).convert("RGB")

        # Bounding box support
        if all(k in row for k in ['x', 'y', 'width', 'height']):
            bbox = [row['x'], row['y'], row['width'], row['height']]
        elif all(k in row for k in ['xmin', 'ymin', 'xmax', 'ymax']):
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            width, height = xmax - xmin, ymax - ymin
            bbox = [xmin, ymin, width, height]
        else:
            bbox = [0, 0, 1, 1]

        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
        return img, bbox_tensor

# ==========================
# PRETRAINED MODEL SETUP
# ==========================
def get_model():
    model = models.swin_t(weights="IMAGENET1K_V1")
    in_features = model.head.in_features
    model.head = torch.nn.Linear(in_features, 4)
    return model

# ==========================
# METRICS: IOU & FROC
# ==========================
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def compute_froc(pred_boxes, true_boxes, iou_threshold=0.5):
    true_positives, false_positives = 0, 0
    for pred, gt in zip(pred_boxes, true_boxes):
        if iou(pred, gt) > iou_threshold:
            true_positives += 1
        else:
            false_positives += 1
    sensitivity = true_positives / max(1, len(true_boxes))
    fps_per_image = false_positives / max(1, len(true_boxes))
    froc_score = sensitivity / (fps_per_image + 1e-5)
    return froc_score, sensitivity, fps_per_image

# ==========================
# TRAINING LOOP
# ==========================
def train_model(dry_run=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wandb.init(project="swin_localization_node21", name="swin_cxr_run")

    transform = transforms.Compose([
        ResizeWithPadding(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_ds = Node21LocalizationDataset(TRAIN_CSV, DATA_ROOT, transform)
    val_ds = Node21LocalizationDataset(VAL_CSV, DATA_ROOT, transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    metrics_records = []

    if dry_run:
        print("Dry run mode: checking loader and transform on batch.")
        for _ in range(2):
            imgs, bboxes = next(iter(train_loader))
            print(f"Batch imgs shape: {imgs.shape}, BBoxes shape: {bboxes.shape}")
        return

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for imgs, bboxes in train_loader:
            imgs, bboxes = imgs.to(device), bboxes.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, bboxes)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        pred_boxes = []
        true_boxes = []
        with torch.no_grad():
            for imgs, bboxes in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs).cpu().numpy()
                pred_boxes.extend(outputs)
                true_boxes.extend(bboxes.numpy())

        froc, sensitiv, fps = compute_froc(pred_boxes, true_boxes)
        wandb.log({
            "epoch": epoch,
            "loss": running_loss,
            "FROC": froc,
            "Sensitivity": sensitiv,
            "FPsPerImage": fps
        })

        metrics_records.append({
            "epoch": epoch,
            "loss": running_loss,
            "FROC": froc,
            "Sensitivity": sensitiv,
            "FPsPerImage": fps
        })

        print(f"Epoch {epoch}: Loss={running_loss:.4f}, FROC={froc:.4f}, Sensitivity={sensitiv:.4f}, FPs/Image={fps:.4f}")

    csv_path = os.path.join(OUTPUT_DIR, "metrics_swin_localization.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metrics_records[0].keys())
        writer.writeheader()
        writer.writerows(metrics_records)

    wandb.finish()

# ==========================
# ENTRY POINT
# ==========================
if __name__ == "__main__":
    train_model(dry_run=DRY_RUN)

