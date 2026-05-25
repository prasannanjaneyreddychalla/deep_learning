import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import autocast, GradScaler

################################################################################
# Paths / Constants
################################################################################

# Training data (same as before – update if needed)
RAW_CSV = '/scratch/smanika3/chexpert/full_uncompressed/train_cheXbert.csv'
TRAIN_ROOT = '/scratch/smanika3/chexpert/full_uncompressed/train'

FILTERED_CSV = 'filtered_chexpert.csv'

# Official CheXpert val & test (paths you showed)
CHEXPERT_ROOT = '/scratch/pchalla7/chexpert/chexlocalize/CheXpert'
VAL_CSV = os.path.join(CHEXPERT_ROOT, 'val_labels.csv')
TEST_CSV = os.path.join(CHEXPERT_ROOT, 'test_labels.csv')
VAL_ROOT = CHEXPERT_ROOT   # expects paths in CSV like "val/patient..."
TEST_ROOT = CHEXPERT_ROOT  # expects paths in CSV like "test/patient..."

batch_size = 32
num_workers = 8
num_epochs = 100

################################################################################
# Step 1: Filter Training CSV and Fix Paths
################################################################################
df = pd.read_csv(RAW_CSV)

# Only keep image files
df = df[df['Path'].str.endswith(('.jpg', '.jpeg', '.png'))]

# Remove incorrect training path prefix if present
df['Path'] = df['Path'].apply(
    lambda x: x.replace('CheXpert-v1.0/train/', '').replace('train/', '')
)
df.to_csv(FILTERED_CSV, index=False)

# Use the entire filtered training set for training
train_df = df.copy()

################################################################################
# Step 2: Compute class weights (for pos_weight in BCEWithLogitsLoss)
################################################################################
# Assumes labels start after column 5. Adjust if needed.
label_cols = train_df.columns[5:]
labels_matrix = train_df[label_cols].values.astype(np.float32)

# Treat -1 (uncertain) as positive for training statistics
labels_matrix_processed = labels_matrix.copy()
labels_matrix_processed[labels_matrix_processed == -1] = 1.0

num_samples, num_classes = labels_matrix_processed.shape
pos_counts = np.sum(labels_matrix_processed == 1, axis=0)
neg_counts = np.sum(labels_matrix_processed == 0, axis=0)

pos_weight_np = np.ones(num_classes, dtype=np.float32)
valid_mask = pos_counts > 0
pos_weight_np[valid_mask] = (neg_counts[valid_mask] / pos_counts[valid_mask]).astype(
    np.float32
)

################################################################################
# Step 3: PyTorch Dataset Definition
################################################################################
class CheXpertDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        # Assumes labels start after column 5. Adjust if needed.
        self.labels = self.data.iloc[:, 5:].values.astype(np.float32)
        # Treat -1 as positive for training / evaluation as well
        self.labels[self.labels == -1] = 1.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_rel = self.data.iloc[idx]['Path']
        img_path = os.path.join(self.root_dir, img_rel)

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Missing file: {img_path}, skipping...")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx])
        return image, label

################################################################################
# Step 4: Transforms and Loaders
################################################################################
# Training: keep 224x224 but use RandomResizedCrop for better augmentation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Validation / Test: deterministic Resize to 224x224
eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Training dataset from filtered CSV and TRAIN_ROOT
train_dataset = CheXpertDataset(FILTERED_CSV, TRAIN_ROOT, transform=train_transforms)

# Official CheXpert val & test sets
val_dataset = CheXpertDataset(VAL_CSV, VAL_ROOT, transform=eval_transforms)
test_dataset = CheXpertDataset(TEST_CSV, TEST_ROOT, transform=eval_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

################################################################################
# Step 5: Model, Loss, Optimizer, Scheduler
################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ConvNeXt-large backbone, 224x224
num_classes = train_dataset.labels.shape[1]
model = timm.create_model('convnext_large', pretrained=True, num_classes=num_classes)
model.to(device)

# Class-balanced BCEWithLogitsLoss using pos_weight
pos_weight = torch.from_numpy(pos_weight_np).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# AdamW optimizer, suited to ConvNeXt
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

# Cosine annealing LR scheduler over epochs
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Mixed-precision scaler
scaler = GradScaler(enabled=(device.type == "cuda"))

################################################################################
# Step 6: Training and Validation Functions
################################################################################
def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    run_loss = 0.0
    preds_all, labels_all = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        run_loss += loss.item() * images.size(0)
        preds_all.append(torch.sigmoid(outputs).detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())

    avg_loss = run_loss / len(loader.dataset)
    preds_all = np.vstack(preds_all)
    labels_all = np.vstack(labels_all)
    try:
        auc = roc_auc_score(labels_all, preds_all, average='macro')
    except ValueError:
        auc = float('nan')
    return avg_loss, auc


def validate_epoch(model, loader, criterion, device):
    model.eval()
    run_loss = 0.0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            run_loss += loss.item() * images.size(0)

            preds_all.append(torch.sigmoid(outputs).cpu().numpy())
            labels_all.append(labels.cpu().numpy())

    avg_loss = run_loss / len(loader.dataset)
    preds_all = np.vstack(preds_all)
    labels_all = np.vstack(labels_all)
    try:
        auc = roc_auc_score(labels_all, preds_all, average='macro')
    except ValueError:
        auc = float('nan')
    return avg_loss, auc

################################################################################
# Step 7: Training Loop (train on TRAIN, validate on official VAL)
################################################################################
train_losses, val_losses = [], []
train_aucs, val_aucs = [], []

best_val_auc = -float('inf')
best_model_path = 'convnext_large_chexpert_best.pth'

for epoch in range(num_epochs):
    train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
    val_loss, val_auc = validate_epoch(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_aucs.append(train_auc)
    val_aucs.append(val_auc)

    scheduler.step()

    print(f'Epoch {epoch+1}/{num_epochs}: '
          f'Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f} | '
          f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), best_model_path)
        print(f'  -> New best model saved with Val AUC {best_val_auc:.4f}')

################################################################################
# Step 8: Final evaluation on official TEST set
################################################################################
test_loss, test_auc = validate_epoch(model, test_loader, criterion, device)
print(f'Final TEST: Loss: {test_loss:.4f}, AUC: {test_auc:.4f}')

################################################################################
# Step 9: Store Results for Future Plotting
################################################################################
results = {
    'epoch': np.arange(1, num_epochs + 1),
    'train_loss': train_losses,
    'val_loss': val_losses,
    'train_auc': train_aucs,
    'val_auc': val_aucs
}
pd.DataFrame(results).to_csv('training_metrics.csv', index=False)
np.savez('training_metrics.npz', **results)

# Final checkpoint (last epoch)
torch.save(model.state_dict(), 'convnext_large_chexpert_last.pth')

