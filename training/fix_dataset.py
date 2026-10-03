import re

# Read the file
with open('training/cnn_improved.py', 'r') as f:
    content = f.read()

# Replace the FER2013Dataset class
old_class = '''class FER2013Dataset(Dataset):
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

        return torch.from_numpy(image), torch.tensor(label, dtype=torch.long)'''

new_class = '''class FER2013Dataset(Dataset):
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

        return image, torch.tensor(self.df.iloc[idx]["emotion"], dtype=torch.long)'''

content = content.replace(old_class, new_class)

with open('training/cnn_improved.py', 'w') as f:
    f.write(content)

print("Done")