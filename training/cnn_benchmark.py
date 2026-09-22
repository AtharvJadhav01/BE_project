"""
CNN Baseline for FER2013

Trains a small CNN directly on 48x48 grayscale FER2013 images.
Uses the exact same train/validation/test split as the MediaPipe benchmarks.
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Label mapping
TARGET_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]
FER2013_LABEL_MAP = {i: name for i, name in enumerate(TARGET_NAMES)}


class FER2013Dataset(Dataset):
    """PyTorch Dataset for FER2013."""
    
    def __init__(self, csv_path, usage_filter=None, transform=None):
        df = pd.read_csv(csv_path)
        if usage_filter:
            df = df[df["Usage"] == usage_filter].copy()
        
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")
        image = pixels.reshape(48, 48).astype(np.float32) / 255.0
        image = image[np.newaxis, :, :]  # Add channel dim: (1, 48, 48)
        label = row["emotion"]
        
        if self.transform:
            image = self.transform(image)
            
        return torch.from_numpy(image), torch.tensor(label, dtype=torch.long)


class SimpleCNN(nn.Module):
    """Small CNN for 48x48 grayscale images."""
    
    def __init__(self, num_classes=7, dropout=0.3):
        super().__init__()
        
        # Conv blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 48 -> 24
            nn.Dropout2d(dropout)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24 -> 12
            nn.Dropout2d(dropout)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 12 -> 6
            nn.Dropout2d(dropout)
        )
        
        # Global average pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


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


def train_cnn():
    """Train CNN on FER2013."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths
    csv_path = Path("data/fer2013.csv")
    
    # Data loaders
    batch_size = 64
    train_dataset = FER2013Dataset(csv_path, usage_filter="Training")
    val_dataset = FER2013Dataset(csv_path, usage_filter="PrivateTest")
    test_dataset = FER2013Dataset(csv_path, usage_filter="PublicTest")
    
    # If PrivateTest is empty, split training
    if len(val_dataset) == 0:
        print("PrivateTest empty, splitting Training for validation")
        train_size = int(0.85 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Model
    model = SimpleCNN(num_classes=7, dropout=0.3).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training loop
    best_val_f1 = 0
    best_model_state = None
    patience = 15
    patience_counter = 0
    max_epochs = 50
    
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(max_epochs):
        # Train
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        val_metrics = evaluate_model(model, val_loader, device)
        val_f1 = val_metrics['f1_macro']
        
        scheduler.step(val_f1)
        
        print(f"Epoch {epoch+1:2d}/{max_epochs}: Train Loss={train_loss:.4f}, Val Acc={val_metrics['accuracy']:.4f}, Val Macro F1={val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    train_time = time.time() - start_time
    
    # Load best model and evaluate on test
    model.load_state_dict(best_model_state)
    test_metrics = evaluate_model(model, test_loader, device)
    
    print(f"\n=== CNN TEST RESULTS ===")
    print(f"Training time: {train_time:.1f}s")
    print(f"Best Val Macro F1: {best_val_f1:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Macro F1: {test_metrics['f1_macro']:.4f}")
    print(f"Test Weighted F1: {test_metrics['f1_weighted']:.4f}")
    print(f"Test Macro Precision: {test_metrics['precision_macro']:.4f}")
    print(f"Test Macro Recall: {test_metrics['recall_macro']:.4f}")
    
    print("\nPer-class Test Results:")
    for cls in TARGET_NAMES:
        m = test_metrics['per_class'][cls]
        print(f"  {cls:10s}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} support={m['support']}")
    
    print("\nConfusion Matrix:")
    cm = np.array(test_metrics['confusion_matrix'])
    header = "      " + "  ".join(f"{n[:4]:>4}" for n in TARGET_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{TARGET_NAMES[i][:4]:>4} " + "  ".join(f"{v:>4}" for v in row))
    
    # Inference latency
    model.eval()
    dummy_input = torch.randn(1, 1, 48, 48).to(device)
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(dummy_input)
        # Measure
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()
        for _ in range(100):
            _ = model(dummy_input)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        infer_time = (time.time() - start) / 100 * 1000
    
    print(f"\nInference latency: {infer_time:.3f} ms/sample")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024:.2f} MB")
    
    return test_metrics, train_time, infer_time


if __name__ == "__main__":
    try:
        metrics, train_time, infer_time = train_cnn()
        
        # Save results
        results = {
            "model": "SimpleCNN",
            "train_time": train_time,
            "inference_ms": infer_time,
            "metrics": metrics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open('training/cnn_benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n[OK] CNN results saved to training/cnn_benchmark_results.json")
        
    except Exception as e:
        print(f"CNN training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save failure info
        with open('training/cnn_benchmark_results.json', 'w') as f:
            json.dump({
                "status": "FAILED",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)