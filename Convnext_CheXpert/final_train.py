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

################################################################################
# Step 1: Filter CSV and Fix Paths
################################################################################
RAW_CSV = '/scratch/smanika3/chexpert/full_uncompressed/train_cheXbert.csv'
FILTERED_CSV = 'filtered_chexpert.csv'

df = pd.read_csv(RAW_CSV)
# Only keep image files
df = df[df['Path'].str.endswith(('.jpg', '.jpeg', '.png'))]
# Remove incorrect path prefix
df['Path'] = df['Path'].apply(lambda x: x.replace('CheXpert-v1.0/train/', ''))
df.to_csv(FILTERED_CSV, index=False)

################################################################################
# Step 2: Split into Train/Val CSVs
################################################################################
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)
train_df.to_csv('train_split.csv', index=False)
val_df.to_csv('val_split.csv', index=False)

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
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

data_root = '/scratch/smanika3/chexpert/full_uncompressed/train'
batch_size = 32

train_dataset = CheXpertDataset('train_split.csv', data_root, transform=train_transforms)
val_dataset = CheXpertDataset('val_split.csv', data_root, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

################################################################################
# Step 5: Model, Loss, Optimizer
################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = train_dataset.labels.shape[1]
model = timm.create_model('convnext_base', pretrained=True, num_classes=num_classes)
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

################################################################################
# Step 6: Training and Validation Functions
################################################################################
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    run_loss = 0.0
    preds_all, labels_all = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
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
            images, labels = images.to(device), labels.to(device)
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
# Step 7: Training Loop
################################################################################
num_epochs = 100
train_losses, val_losses = [], []
train_aucs, val_aucs = [], []

for epoch in range(num_epochs):
    train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_auc = validate_epoch(model, val_loader, criterion, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_aucs.append(train_auc)
    val_aucs.append(val_auc)
    print(f'Epoch {epoch+1}/{num_epochs}: '
          f'Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f} | '
          f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')

################################################################################
# Step 8: Store Results for Future Plotting
################################################################################
results = {
    'epoch': np.arange(1, num_epochs+1),
    'train_loss': train_losses,
    'val_loss': val_losses,
    'train_auc': train_aucs,
    'val_auc': val_aucs
}
pd.DataFrame(results).to_csv('training_metrics.csv', index=False)
np.savez('training_metrics.npz', **results)
torch.save(model.state_dict(), 'convnext_chexpert.pth')

