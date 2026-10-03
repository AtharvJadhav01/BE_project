"""
CNN Training for FER2013 - Exact Benchmark Protocol
Uses the verified 1,162 face-detected PublicTest samples from cnn_benchmark_split.json
"""

import sys
import time
import json
import random
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

# Reproducibility seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

TARGET_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]


class FER2013Dataset(Dataset):
    """PyTorch Dataset for FER2013 raw images with optional augmentation."""

    def __init__(self, df, augment=False):
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

        if self.augment:
            # Random horizontal flip (50% chance)
            if random.random() > 0.5:
                img = torch.flip(img, dims=[2])
            # Small random rotation (-10 to 10 degrees)
            angle = random.uniform(-10, 10)
            # Small random translation (-2 to 2 pixels)
            tx = random.uniform(-2, 2)
            ty = random.uniform(-2, 2)
            img = TF.affine(img, angle=angle, translate=(tx/48, ty/48), scale=1.0, shear=0)

        return img, label


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
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_benchmark_split():
    """Load the exact 1,162 face-detected test indices from cnn_benchmark_split.json"""
    with open('training/cnn_benchmark_split.json', 'r') as f:
        split_data = json.load(f)
    
    face_detected_indices = split_data['face_detected_csv_indices']
    labels = split_data['labels']
    
    print(f"Loaded benchmark split: {len(face_detected_indices)} test samples")
    print(f"Class distribution: {split_data['class_distribution']}")
    
    return face_detected_indices, labels


def prepare_datasets():
    """Prepare train/val/test datasets following exact protocol"""
    df = pd.read_csv('data/fer2013.csv')
    
    # Load benchmark split
    test_csv_indices, test_labels = load_benchmark_split()
    
    # Create test DataFrame from exact indices (preserve original CSV indices)
    test_df = df.loc[test_csv_indices].copy()
    print(f"Test set: {len(test_df)} samples")
    
    # Verify test labels match
    test_df_labels = test_df['emotion'].tolist()
    assert test_df_labels == test_labels, "Test labels mismatch!"
    print("Test labels verified against benchmark split.")
    
    # Training data: only Usage == "Training"
    train_all_df = df[df['Usage'] == 'Training'].copy()
    print(f"Training pool: {len(train_all_df)} samples")
    
    # 85/15 train/val split from Training, stratified
    train_df, val_df = train_test_split(
        train_all_df, test_size=0.15, random_state=SEED, stratify=train_all_df['emotion']
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Verify no overlap (check original CSV indices)
    train_csv_indices = set(train_df.index)
    val_csv_indices = set(val_df.index)
    test_csv_indices = set(test_df.index)
    assert len(train_csv_indices & val_csv_indices) == 0
    assert len(train_csv_indices & test_csv_indices) == 0
    assert len(val_csv_indices & test_csv_indices) == 0
    print("No overlap between train/val/test verified.")
    
    # Create datasets
    train_dataset = FER2013Dataset(train_df, augment=True)
    val_dataset = FER2013Dataset(val_df, augment=False)
    test_dataset = FER2013Dataset(test_df, augment=False)
    
    return train_dataset, val_dataset, test_dataset


def train():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    
    # Data
    train_dataset, val_dataset, test_dataset = prepare_datasets()
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Model
    model = SimpleCNN(num_classes=7, dropout=0.3).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {param_count:,}')
    
    # Class weights for imbalance (disgust is underrepresented in test)
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # OneCycleLR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    # Training
    best_val_f1 = 0
    best_state = None
    patience = 15
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'lr': []
    }
    
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(50):
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
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1:2d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Acc={val_acc:.4f}, Val Macro F1={val_f1:.4f}, LR={current_lr:.6f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        scheduler.step()
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (best epoch: {best_epoch})")
                break
    
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.1f}s")
    print(f"Best epoch: {best_epoch}, Best Val Macro F1: {best_val_f1:.4f}")
    
    # Load best model and evaluate on test
    model.load_state_dict(best_state)
    model.eval()
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())
    
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n=== FINAL TEST RESULTS ===")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Macro F1: {macro_f1:.4f} ({macro_f1*100:.2f}%)")
    print(f"Weighted F1: {weighted_f1:.4f} ({weighted_f1*100:.2f}%)")
    
    # Per-class
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    per_class = {}
    for i, cls in enumerate(TARGET_NAMES):
        per_class[cls] = {
            'precision': float(prec[i]),
            'recall': float(rec[i]),
            'f1': float(f1[i]),
            'support': int(np.sum(np.array(y_true) == i))
        }
        print(f"  {cls:10s}: P={prec[i]:.4f} R={rec[i]:.4f} F1={f1[i]:.4f} support={per_class[cls]['support']}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save model checkpoint
    torch.save(model.state_dict(), 'training/models/cnn_fer2013_best.pt')
    print("\nModel saved to training/models/cnn_fer2013_best.pt")
    
    # Save metadata
    metadata = {
        'architecture': 'SimpleCNN',
        'parameter_count': param_count,
        'input_size': [1, 48, 48],
        'num_classes': 7,
        'class_names': TARGET_NAMES,
        'dataset_counts': {
            'train': len(train_dataset),
            'val': len(val_dataset),
            'test': len(test_dataset)
        },
        'benchmark_split_path': 'training/cnn_benchmark_split.json',
        'seed': SEED,
        'training_config': {
            'optimizer': 'AdamW',
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'scheduler': 'OneCycleLR',
            'max_lr': 1e-2,
            'pct_start': 0.3,
            'batch_size': 64,
            'max_epochs': 50,
            'early_stopping_patience': 15,
            'class_weights': [1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            'augmentation': 'horizontal_flip, rotation(-10,10), translation(-2,2)'
        },
        'best_epoch': best_epoch,
        'best_val_macro_f1': float(best_val_f1),
        'test_metrics': {
            'accuracy': float(accuracy),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'per_class': per_class,
            'confusion_matrix': cm.tolist()
        },
        'linear_svc_benchmark': {
            'accuracy': 0.5069,
            'macro_f1': 0.4919,
            'weighted_f1': 0.4998
        },
        'training_time_seconds': train_time
    }
    
    with open('training/models/cnn_fer2013_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Metadata saved to training/models/cnn_fer2013_metadata.json")
    
    # Save history
    with open('training/models/cnn_fer2013_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("History saved to training/models/cnn_fer2013_history.json")
    
    # Reload verification
    print("\n=== RELOAD VERIFICATION ===")
    model2 = SimpleCNN(num_classes=7, dropout=0.3).to(device)
    model2.load_state_dict(torch.load('training/models/cnn_fer2013_best.pt', map_location=device))
    model2.eval()
    
    y_pred2 = []
    y_true2 = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model2(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred2.extend(preds)
            y_true2.extend(labels.numpy())
    
    preds_match = np.array_equal(y_pred, y_pred2)
    print(f"Predictions identical after reload: {preds_match}")
    
    if preds_match:
        acc2 = accuracy_score(y_true2, y_pred2)
        macro_f1_2 = f1_score(y_true2, y_pred2, average='macro')
        weighted_f1_2 = f1_score(y_true2, y_pred2, average='weighted')
        print(f"Reload metrics match: Acc={acc2:.4f}, MacroF1={macro_f1_2:.4f}, WeightedF1={weighted_f1_2:.4f}")
    
    return metadata, history, preds_match


if __name__ == '__main__':
    metadata, history, reload_ok = train()
    print(f"\n=== SUMMARY ===")
    print(f"Training completed: True")
    print(f"Best epoch: {metadata['best_epoch']}")
    print(f"Best val Macro F1: {metadata['best_val_macro_f1']:.4f}")
    print(f"Test Accuracy: {metadata['test_metrics']['accuracy']:.4f}")
    print(f"Macro F1: {metadata['test_metrics']['macro_f1']:.4f}")
    print(f"Weighted F1: {metadata['test_metrics']['weighted_f1']:.4f}")
    print(f"Parameter count: {metadata['parameter_count']:,}")
    print(f"Checkpoint path: training/models/cnn_fer2013_best.pt")
    print(f"Reload verification: {'PASSED' if reload_ok else 'FAILED'}")