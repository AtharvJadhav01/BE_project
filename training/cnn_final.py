"""
Simple CNN for FER2013 - Optimized for convergence
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Label mapping
TARGET_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]


class FER2013Dataset(Dataset):
    """PyTorch Dataset for FER2013 raw images."""

    def __init__(self, csv_path, usage_filter=None, augment=False):
        df = pd.read_csv(csv_path)
        if usage_filter:
            df = df[df["Usage"] == usage_filter].copy()

        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")
        image = pixels.reshape(48, 48).astype(np.float32) / 255.0
        image = image[np.newaxis, :, :]  # (1, 48, 48)
        img = torch.from_numpy(image).float()
        label = torch.tensor(row["emotion"], dtype=torch.long)
        return img, label

    @classmethod
    def from_dataframe(cls, df, augment=False):
        """Create a FER2013Dataset from an existing DataFrame (already filtered)."""
        instance = object.__new__(cls)
        instance.df = df.reset_index(drop=True)
        instance.augment = augment
        return instance


class SimpleCNN(nn.Module):
    """Simple, well-regularized CNN for 48x48 grayscale images."""

    def __init__(self, num_classes=7, dropout=0.3):
        super().__init__()

        # 48x48 -> 24x24
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 48 -> 24
            nn.Dropout2d(0.25)
        )

        # 24 -> 12
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24 -> 12
            nn.Dropout2d(0.25)
        )

        # 12 -> 6
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 12 -> 6
            nn.Dropout2d(0.25)
        )

        # 128 -> 6x6 = 4608 -> 256
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 7)
        )

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
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_data():
    """Load FER2013 data and create train/val/test datasets."""
    df = pd.read_csv('data/fer2013.csv')
    
    train_df = df[df['Usage'] == 'Training'].copy()
    test_df = df[df['Usage'] == 'PublicTest'].copy()
    
    # 85/15 train/val split from Training
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df['emotion'])
    test_df = test_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    train_dataset = FER2013Dataset.from_dataframe(train_df, augment=True)
    val_dataset = FER2013Dataset.from_dataframe(val_df, augment=False)
    test_dataset = FER2013Dataset.from_dataframe(test_df, augment=False)
    
    return train_dataset, val_dataset, test_dataset


class SimpleCNN(nn.Module):
    """Simple, well-regularized CNN for 48x48 grayscale images."""

    def __init__(self, num_classes=7, dropout=0.3):
        super().__init__()

        # 48x48 -> 24x24
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 48 -> 24
            nn.Dropout2d(0.25)
        )

        # 24 -> 12
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24 -> 12
            nn.Dropout2d(0.25)
        )

        # 12 -> 6
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 12 -> 6
            nn.Dropout2d(0.25)
        )

        # 128 -> 6x6 = 4608 -> 256
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 7)
        )

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
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    
    # Load data
    train_dataset, val_dataset, test_dataset = load_data()
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Model
    model = SimpleCNN(num_classes=7, dropout=0.3).cuda()
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Class weights for imbalance (disgust is underrepresented)
    class_weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights.cuda())
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, epochs=50, steps_per_epoch=100, pct_start=0.3)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    best_val_f1 = 0
    best_state = None
    patience = 15
    patience_counter = 0
    
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(50):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
        
        train_loss /= len(train_loader)
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                val_loss += loss.item()
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        from sklearn.metrics import f1_score
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        
        print(f"Epoch {epoch+1:2d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Val Macro F1={val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best and evaluate on test
    model.load_state_dict(best_state)
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())
    
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Macro F1: {macro_f1:.4f} ({macro_f1*100:.2f}%)")
    print(f"Weighted F1: {weighted_f1:.4f} ({weighted_f1*100:.2f}%)")
    
    # Per-class
    from sklearn.metrics import precision_score, recall_score
    prec = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
    for i, cls in enumerate(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']):
        print(f"  {cls:10s}: P={prec[i]:.4f} R={rec[i]:.4f} F1={f1[i]:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'training/models/cnn_fer2013_best.pt')
    print("\nModel saved to training/models/cnn_fer2013_best.pt")

if __name__ == '__main__':
    main()