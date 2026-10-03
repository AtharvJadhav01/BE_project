"""
Improved CNN for FER2013 - Fixed training issues
"""

import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Label mapping
TARGET_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]
FER2013_LABEL_MAP = {i: name for i, name in enumerate(TARGET_NAMES)}


class FER2013Dataset(Dataset):
    """PyTorch Dataset for FER2013."""

    def __init__(self, csv_path, usage_filter=None, transform=None, augment=False):
        df = pd.read_csv(csv_path)
        if usage_filter:
            df = df[df["Usage"] == usage_filter].copy()

        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")
        image = pixels.reshape(48, 48).astype(np.float32) / 255.0
        image = image[np.newaxis, :, :]  # Add channel dim: (1, 48, 48)
        label = row["emotion"]

        # Convert to tensor
        image = torch.from_numpy(image)

        # Apply augmentations during training
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                image = torch.flip(image, dims=[2])
            # Small random rotation (-10 to 10 degrees)
            angle = np.random.uniform(-10, 10)
            # Simple translation
            tx = np.random.uniform(-2, 2)
            ty = np.random.uniform(-2, 2)
            # Apply affine transform
            import torchvision.transforms.functional as TF
            image = TF.affine(image, angle=angle, translate=(tx/48, ty/48), scale=1.0, shear=0)

        if self.transform and not self.augment:
            image = self.transform(image)

        return image, torch.tensor(self.df.iloc[idx]["emotion"], dtype=torch.long)


class SimpleCNN(nn.Module):
    """Simpler CNN for 48x48 grayscale images - less prone to overfitting."""

    def __init__(self, num_classes=7, dropout=0.25):
        super().__init__()

        # Conv blocks - fewer layers, wider
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 48 -> 24
            nn.Dropout2d(0.25)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24 -> 12
            nn.Dropout2d(0.25)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 12 -> 6
            nn.Dropout2d(0.25)
        )

        # Global average pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 7)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


class ImprovedCNN(nn.Module):
    """Improved CNN with residual connections and better regularization."""

    def __init__(self, num_classes=7, dropout=0.3):
        super().__init__()

        # Initial conv
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.layer1 = self._make_layer(32, 64, stride=2, dropout=0.2)  # 48->24
        self.layer2 = self._make_layer(64, 128, stride=2, dropout=0.3)  # 24->12
        self.layer3 = self._make_layer(128, 256, stride=2, dropout=0.4)  # 12->6

        # Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 7)
        )

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, stride, dropout):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


class FER2013Dataset(Dataset):
    """PyTorch Dataset for FER2013."""

    def __init__(self, csv_path, usage_filter=None, transform=None, augment=False):
        df = pd.read_csv(csv_path)
        if usage_filter:
            df = df[df["Usage"] == usage_filter].copy()

        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")
        image = pixels.reshape(48, 48).astype(np.float32) / 255.0
        image = image[np.newaxis, :, :]  # Add channel dim: (1, 48, 48)
        label = row["emotion"]

        # Convert to tensor
        image = torch.from_numpy(image)

        # Apply augmentations during training
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                image = torch.flip(image, dims=[2])
            # Small random rotation (-10 to 10 degrees)
            angle = np.random.uniform(-10, 10)
            # Simple translation
            tx = np.random.uniform(-2, 2)
            ty = np.random.uniform(-2, 2)
            # Apply affine transform
            import torchvision.transforms.functional as TF
            image = TF.affine(image, angle=angle, translate=(tx/48, ty/48), scale=1.0, shear=0)

        if self.transform and not self.augment:
            image = self.transform(image)

        return image, torch.tensor(self.df.iloc[idx]["emotion"], dtype=torch.long)


def evaluate_model(model, dataloader, device):
    """Evaluate model on dataloader."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    precision_w = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall_w = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1_w = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    cm = confusion_matrix(all_labels, all_preds)

    per_class = {}
    for i, name in enumerate(TARGET_NAMES):
        mask = np.array(all_labels) == i
        if np.sum(mask) > 0:
            per_class[name] = {
                "precision": precision_score(all_labels, all_preds, labels=[i], average="macro", zero_division=0),
                "recall": recall_score(all_labels, all_preds, labels=[i], average="macro", zero_division=0),
                "f1": f1_score(all_labels, all_preds, labels=[i], average="macro", zero_division=0),
                "support": int(np.sum(mask))
            }
        else:
            per_class[name] = {"precision": 0, "recall": 0, "f1": 0, "support": 0}

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_w),
        "recall_weighted": float(recall_w),
        "f1_weighted": float(f1_w),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class
    }


# Data augmentation
class AugmentedFER2013Dataset(FER2013Dataset):
    """FER2013 dataset with augmentation for training."""

    def __init__(self, csv_path, usage_filter=None, transform=None, augment=False):
        super().__init__(csv_path, usage_filter, transform)
        self.augment = augment

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")
        image = pixels.reshape(48, 48).astype(np.float32) / 255.0
        image = image[np.newaxis, :, :]  # (1, 48, 48)
        label = row["emotion"]

        # Convert to tensor
        image = torch.from_numpy(image)

        # Apply augmentations during training
        if self.augment and self.transform:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                image = torch.flip(image, dims=[2])
            # Small random rotation (-10 to 10 degrees)
            angle = np.random.uniform(-10, 10)
            # Simple translation
            tx = np.random.uniform(-2, 2)
            ty = np.random.uniform(-2, 2)
            # Apply affine transform
            from torchvision import transforms
            import torchvision.transforms.functional as TF
            image = TF.affine(image, angle=angle, translate=(tx/48, ty/48), scale=1.0, shear=0)

        if self.transform and not self.augment:
            image = self.transform(image)

        return image, torch.tensor(self.df.iloc[idx]["emotion"], dtype=torch.long)


def train_model(model, train_loader, val_loader, test_loader, device, epochs=50, patience=15):
    """Train model with proper early stopping and learning rate scheduling."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Class weights for imbalanced classes (from validation set analysis)
    # Class distribution in test: angry:200, disgust:111, fear:200, happy:200, sad:200, surprised:200, neutral:200
    # But validation set has different distribution - use balanced weights
    class_weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, epochs=50, steps_per_epoch=100,
        pct_start=0.3, div_factor=10, final_div_factor=100
    )

    best_val_f1 = 0
    best_model_state = None
    patience = 15
    patience_counter = 0

    print(f"\nStarting training on {device}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")

    start_time = time.time()

    for epoch in range(60):
        # Train
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = correct / total

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        scheduler.step()

        print(f"Epoch {epoch+1:2d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val Macro F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"Early stopping at epoch {epoch+1}")
                break

    train_time = time.time() - start_time

    # Load best model and evaluate on test
    model.load_state_dict(best_model_state)
    test_metrics = evaluate_model(model, test_loader, device)

    return test_metrics, train_time


def train_cnn_improved():
    """Train improved CNN on FER2013."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data paths
    csv_path = Path("data/fer2013.csv")

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])

    # No augmentation for validation/test
    val_transform = None

    # Load datasets
    train_dataset = FER2013Dataset("data/fer2013.csv", usage_filter="Training", transform=None, augment=True)
    val_dataset = FER2013Dataset("data/fer2013.csv", usage_filter="PrivateTest", transform=None)
    test_dataset = FER2013Dataset("data/fer2013.csv", usage_filter="PublicTest", transform=None)

    # Split training data for validation since PrivateTest is empty
    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    # Enable augmentation for training subset
    train_dataset.dataset.augment = True

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Model
    model = ImprovedCNN(num_classes=7, dropout=0.3).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights
    class_weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, epochs=60, steps_per_epoch=len(train_loader),
        pct_start=0.3, div_factor=10, final_div_factor=100
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    return train_model(ImprovedCNN(num_classes=7, dropout=0.3), train_loader, val_loader, test_loader, device)


if __name__ == "__main__":
    try:
        metrics = train_cnn_improved()
        print(f"\n=== FINAL RESULTS ===")
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test Macro F1: {metrics['f1_macro']:.4f}")
        print(f"Test Weighted F1: {metrics['f1_weighted']:.4f}")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()