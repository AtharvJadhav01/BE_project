"""
Webcam Expression Dataset Validator

Validates the collected webcam expression dataset.
Checks for corrupted images, missing files, invalid labels, duplicates,
image dimensions, face detection, class distribution, and metadata.
"""

import os
import cv2
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from face.detector import FaceDetector, FaceDetectionResult


EXPRESSION_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprised",
    "neutral",
]


class WebcamDatasetValidator:
    """Validates webcam expression dataset."""

    def __init__(self, dataset_dir: str = "data/webcam_expression_dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.metadata_file = self.dataset_dir / "metadata.jsonl"
        self.detector = FaceDetector({
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
        })

    def validate(self) -> Dict:
        """Run all validation checks and return summary."""
        print("=== Webcam Expression Dataset Validator ===\n")

        if not self.dataset_dir.exists():
            print(f"Dataset directory does not exist: {self.dataset_dir}")
            return {"status": "DATASET NOT YET COLLECTED"}

        # Find all image files
        image_files = self._find_image_files()
        if not image_files:
            print("No image files found in dataset directory")
            return {"status": "DATASET NOT YET COLLECTED"}

        print(f"Found {len(image_files)} image files\n")

        # Run all checks
        results = {}
        results["total_images"] = len(image_files)
        results["corrupted"] = self._check_corrupted_images(image_files)
        results["duplicates"] = self._check_duplicates(image_files)
        results["dimensions"] = self._check_dimensions(image_files)
        results["face_detection"] = self._check_face_detection(image_files)
        results["class_distribution"] = self._check_class_distribution(image_files)
        results["metadata"] = self._check_metadata(image_files)
        results["labels"] = self._check_labels(image_files)

        # Print summary
        self._print_summary(results)
        return results

    def _find_image_files(self) -> List[Path]:
        """Find all image files in class directories."""
        files = []
        for cls in EXPRESSION_CLASSES:
            cls_dir = self.dataset_dir / cls
            if cls_dir.exists():
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                    files.extend(cls_dir.glob(ext))
        return files

    def _check_corrupted_images(self, image_files: List[Path]) -> Dict:
        """Check for corrupted/unreadable images."""
        print("Checking for corrupted images...")
        corrupted = []
        for f in image_files:
            img = cv2.imread(str(f))
            if img is None:
                corrupted.append(str(f))
        print(f"  Corrupted: {len(corrupted)}")
        return {"count": len(corrupted), "files": corrupted}

    def _check_duplicates(self, image_files: List[Path]) -> Dict:
        """Check for duplicate images using content hash."""
        print("Checking for duplicate images...")
        hashes: Dict[str, Path] = {}
        duplicates = []
        for f in image_files:
            try:
                with open(f, "rb") as fp:
                    file_hash = hashlib.md5(fp.read()).hexdigest()
                if file_hash in hashes:
                    duplicates.append({"file": str(f), "duplicate_of": str(hashes[file_hash])})
                else:
                    hashes[file_hash] = f
            except Exception:
                pass
        print(f"  Duplicates: {len(duplicates)}")
        return {"count": len(duplicates), "pairs": duplicates}

    def _check_dimensions(self, image_files: List[Path]) -> Dict:
        """Check image dimensions consistency."""
        print("Checking image dimensions...")
        dims = Counter()
        for f in image_files:
            img = cv2.imread(str(f))
            if img is not None:
                h, w = img.shape[:2]
                dims[(w, h)] += 1
        print(f"  Unique dimensions: {len(dims)}")
        for (w, h), count in dims.most_common():
            print(f"    {w}x{h}: {count} images")
        return {"count": len(dims), "distribution": dict(dims)}

    def _check_face_detection(self, image_files: List[Path]) -> Dict:
        """Run face detection on all images."""
        print("Running face detection...")
        detected = 0
        not_detected = 0
        multiple_faces = 0  # Not directly available, but we can note single-face assumption
        detection_confidences = []

        for i, f in enumerate(image_files):
            if i % 50 == 0 and i > 0:
                print(f"  Processed {i}/{len(image_files)}...")

            img = cv2.imread(str(f))
            if img is None:
                not_detected += 1
                continue

            result = self.detector.process_frame(img)
            if result.face_detected:
                detected += 1
                detection_confidences.append(result.detection_confidence)
            else:
                not_detected += 1

        print(f"  Face detected: {detected}")
        print(f"  No face detected: {not_detected}")
        if detection_confidences:
            avg_conf = sum(detection_confidences) / len(detection_confidences)
            min_conf = min(detection_confidences)
            print(f"  Avg detection confidence: {avg_conf:.3f}")
            print(f"  Min detection confidence: {min_conf:.3f}")

        return {
            "detected": detected,
            "not_detected": not_detected,
            "detection_rate": detected / len(image_files) if image_files else 0,
            "avg_confidence": sum(detection_confidences) / len(detection_confidences) if detection_confidences else 0,
            "min_confidence": min(detection_confidences) if detection_confidences else 0,
        }

    def _check_class_distribution(self, image_files: List[Path]) -> Dict:
        """Check class distribution."""
        print("Checking class distribution...")
        class_counts = Counter()
        for f in image_files:
            # Class is the parent directory name
            cls = f.parent.name
            if cls in EXPRESSION_CLASSES:
                class_counts[cls] += 1

        print(f"  Class distribution:")
        for cls in EXPRESSION_CLASSES:
            count = class_counts.get(cls, 0)
            print(f"    {cls}: {count}")

        total = sum(class_counts.values())
        if total > 0:
            min_cls = min(class_counts, key=class_counts.get)
            max_cls = max(class_counts, key=class_counts.get)
            imbalance_ratio = class_counts[max_cls] / class_counts[min_cls] if class_counts[min_cls] > 0 else float('inf')
            print(f"  Imbalance ratio (max/min): {imbalance_ratio:.1f}")

        return {"counts": dict(class_counts), "total": total}

    def _check_labels(self, image_files: List[Path]) -> Dict:
        """Check for invalid labels."""
        print("Checking labels...")
        invalid = []
        for f in image_files:
            cls = f.parent.name
            if cls not in EXPRESSION_CLASSES:
                invalid.append({"file": str(f), "found_label": cls})
        print(f"  Invalid labels: {len(invalid)}")
        return {"count": len(invalid), "invalid": invalid}

    def _check_metadata(self, image_files: List[Path]) -> Dict:
        """Check metadata file existence and completeness."""
        print("Checking metadata...")
        if not self.metadata_file.exists():
            print("  Metadata file not found: metadata.jsonl")
            return {"exists": False, "entries": 0, "missing_from_metadata": len(image_files)}

        metadata_entries = []
        image_paths_in_metadata = set()

        try:
            with open(self.metadata_file, "r") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        metadata_entries.append(entry)
                        if "image_path" in entry:
                            # Normalize path
                            img_path = Path(entry["image_path"])
                            image_paths_in_metadata.add(str(img_path))
                    except json.JSONDecodeError as e:
                        print(f"  Warning: Invalid JSON at line {line_num}: {e}")
        except Exception as e:
            print(f"  Error reading metadata: {e}")
            return {"exists": True, "entries": 0, "error": str(e)}

        # Check coverage
        actual_image_paths = {str(f.relative_to(self.dataset_dir)) for f in image_files}
        missing_from_metadata = actual_image_paths - image_paths_in_metadata
        extra_in_metadata = image_paths_in_metadata - actual_image_paths

        print(f"  Metadata entries: {len(metadata_entries)}")
        print(f"  Images missing from metadata: {len(missing_from_metadata)}")
        print(f"  Extra entries in metadata: {len(extra_in_metadata)}")

        return {
            "exists": True,
            "entries": len(metadata_entries),
            "missing_from_metadata": len(missing_from_metadata),
            "extra_in_metadata": len(extra_in_metadata),
            "missing_files": list(missing_from_metadata)[:10],  # Show first 10
        }

    def _print_summary(self, results: Dict):
        """Print validation summary."""
        print("\n" + "=" * 50)
        print("VALIDATION SUMMARY")
        print("=" * 50)

        if results.get("total_images", 0) == 0:
            print("DATASET NOT YET COLLECTED")
            return

        print(f"Total images: {results['total_images']}")
        print(f"Corrupted images: {results['corrupted']['count']}")
        print(f"Duplicate images: {results['duplicates']['count']}")
        print(f"Unique dimensions: {results['dimensions']['count']}")
        print(f"Face detection rate: {results['face_detection']['detection_rate']:.1%}")
        print(f"Avg detection confidence: {results['face_detection']['avg_confidence']:.3f}")

        class_dist = results['class_distribution']
        if class_dist['total'] > 0:
            print(f"Class distribution:")
            for cls in EXPRESSION_CLASSES:
                count = class_dist['counts'].get(cls, 0)
                pct = count / class_dist['total'] * 100
                print(f"  {cls}: {count} ({pct:.1f}%)")

        if results['metadata']['exists']:
            print(f"Metadata coverage: {results['metadata']['entries']} entries")
            print(f"  Missing from metadata: {results['metadata']['missing_from_metadata']}")
        else:
            print("Metadata: NOT FOUND")

        print("=" * 50)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate webcam expression dataset")
    parser.add_argument(
        "--dataset",
        default="data/webcam_expression_dataset",
        help="Dataset directory",
    )
    args = parser.parse_args()

    validator = WebcamDatasetValidator(args.dataset)
    validator.validate()


if __name__ == "__main__":
    main()