"""
Convert FER2013 image folder structure to CSV format.
"""

import os
import sys
import pandas as pd
from pathlib import Path
import argparse

# Label mapping (folder name -> label index)
LABEL_MAP = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6
}

LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


def convert_folder_to_csv(data_dir: Path, output_csv: Path):
    """Convert folder structure to FER2013 CSV format."""
    
    rows = []
    
    # Check both train and test directories
    for split_name in ["train", "test"]:
        split_dir = data_dir / split_name
        if not split_dir.exists():
            print(f"Split directory not found: {split_dir}")
            continue
            
        print(f"Processing {split_name} split...")
        
        for emotion_folder in split_dir.iterdir():
            if not emotion_folder.is_dir():
                continue
                
            emotion = emotion_folder.name
            if emotion not in LABEL_MAP:
                print(f"Unknown emotion folder: {emotion}")
                continue
            
            label_idx = LABEL_MAP[emotion]
            
            # Count images
            image_files = list(emotion_folder.glob("*.jpg")) + list(emotion_folder.glob("*.png"))
            print(f"  {emotion}: {len(image_files)} images")
            
            for img_path in image_files:
                # Read image and convert to pixel string
                import cv2
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Failed to read: {img_path}")
                    continue
                
                # Resize to 48x48 if needed
                if img.shape != (48, 48):
                    img = cv2.resize(img, (48, 48))
                
                # Convert to pixel string
                pixels = " ".join(map(str, img.flatten()))
                
                rows.append({
                    "emotion": LABEL_MAP[emotion],
                    "pixels": pixels,
                    "Usage": "Training" if split_name == "train" else "PublicTest"
                })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save CSV
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} samples to {output_csv}")
    
    # Print statistics
    print("\nClass distribution:")
    for idx, name in sorted(LABEL_MAP.items(), key=lambda x: x[1]):
        count = len(df[df["emotion"] == idx])
        print(f"  {name}: {count}")
    
    print("\nSplit distribution:")
    print(df["Usage"].value_counts())


def main():
    parser = argparse.ArgumentParser(description="Convert FER2013 folder structure to CSV")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory with train/test folders")
    parser.add_argument("--output", type=str, default="data/fer2013.csv", help="Output CSV path")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_csv = Path(args.output)
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    convert_folder_to_csv(data_dir, output_csv)


if __name__ == "__main__":
    import cv2  # Import here
    main()