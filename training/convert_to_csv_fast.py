"""
Convert FER2013 image folder structure to CSV format - fast version with multiprocessing.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict
import time

# Label mapping
LABEL_MAP = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6
}

USAGE_MAP = {
    "train": "Training",
    "test": "PublicTest"
}


def process_image(img_path: str, label_idx: int, usage: str) -> dict:
    """Process a single image file."""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        if img.shape != (48, 48):
            img = cv2.resize(img, (48, 48))
        
        pixels = " ".join(map(str, img.flatten()))
        
        return {
            "emotion": label_idx,
            "pixels": pixels,
            "Usage": usage
        }
    except Exception as e:
        return None


def process_emotion_folder(args: Tuple[str, int, str, int]) -> List[dict]:
    """Process all images in an emotion folder."""
    emotion_folder_str, label_idx, usage, max_per_class = args
    
    emotion_folder = Path(emotion_folder_str)
    results = []
    image_files = list(emotion_folder.glob("*.jpg")) + list(emotion_folder.glob("*.png"))
    
    # Limit per class if specified
    if max_per_class and len(image_files) > max_per_class:
        image_files = image_files[:max_per_class]
    
    print(f"  [Worker] Processing {emotion_folder.name} with label_idx={label_idx}, {len(image_files)} images")
    
    for img_path in image_files:
        result = process_image(str(img_path), label_idx, usage)
        if result:
            # Verify the label is correct
            if result["emotion"] != label_idx:
                print(f"  WARNING: Label mismatch! Expected {label_idx}, got {result['emotion']}")
            results.append(result)
    
    return results


def convert_folder_to_csv(data_dir: str, output_csv: str, max_per_class: int = None, n_workers: int = 4):
    """Convert folder structure to FER2013 CSV format using multiprocessing."""
    
    data_dir = Path(data_dir)
    output_csv = Path(output_csv)
    
    all_rows = []
    
    # Prepare tasks
    tasks = []
    for split_name in ["train", "test"]:
        split_dir = Path(data_dir) / split_name
        if not split_dir.exists():
            print(f"Split directory not found: {split_dir}")
            continue
            
        print(f"Processing {split_name} split...")
        
        for emotion_folder in split_dir.iterdir():
            if not emotion_folder.is_dir():
                continue
                
            emotion = emotion_folder.name
            if emotion not in LABEL_MAP:
                print(f"Unknown emotion: {emotion}")
                continue
            
            label_idx = LABEL_MAP[emotion]
            usage = USAGE_MAP.get(split_name, "PublicTest")
            
            print(f"  Found {emotion} -> label {label_idx}")
            
            tasks.append((str(emotion_folder), label_idx, usage, max_per_class))
    
    print(f"Processing {len(tasks)} emotion folders with {n_workers} workers...")
    
    # Process in parallel
    all_rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_emotion_folder, task) for task in tasks]
        
        for future in as_completed(futures):
            try:
                results = future.result()
                if results:
                    print(f"  Got {len(results)} results with emotions: {set(r['emotion'] for r in results)}")
                all_rows.extend(results)
            except Exception as e:
                print(f"Error processing folder: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(all_rows)
    
    if len(df) == 0:
        print("No data processed!")
        return
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save CSV
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert FER2013 folder structure to CSV (fast)")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory with train/test folders")
    parser.add_argument("--output", type=str, default="data/fer2013.csv", help="Output CSV path")
    parser.add_argument("--max-per-class", type=int, default=None, help="Max samples per class (for testing)")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    
    args = parser.parse_args()
    
    start_time = time.time()
    convert_folder_to_csv(args.data_dir, args.output, args.max_per_class, args.workers)
    print(f"\nTotal time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()