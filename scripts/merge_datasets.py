"""
Merge Road Lane and BDD100K datasets into a unified dataset with 17 classes.

This script:
1. Copies Road Lane images and labels (class IDs 0-5 unchanged)
2. Copies BDD100K images and remaps class IDs:
   - Skips BDD100K's generic 'lane' class (original ID 4)
   - Remaps remaining classes to IDs 6-16

Class Mapping:
--------------
Unified ID | Class Name      | Source
-----------|-----------------|--------
0          | divider-line    | Road Lane
1          | dotted-line     | Road Lane
2          | double-line     | Road Lane
3          | random-line     | Road Lane
4          | road-sign-line  | Road Lane
5          | solid-line      | Road Lane
6          | bike            | BDD100K (was 0)
7          | bus             | BDD100K (was 1)
8          | car             | BDD100K (was 2)
9          | drivable area   | BDD100K (was 3)
10         | motor           | BDD100K (was 5)
11         | person          | BDD100K (was 6)
12         | rider           | BDD100K (was 7)
13         | traffic light   | BDD100K (was 8)
14         | traffic sign    | BDD100K (was 9)
15         | train           | BDD100K (was 10)
16         | truck           | BDD100K (was 11)

Usage:
    python merge_datasets.py
    python merge_datasets.py --validate-only
"""
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from collections import defaultdict

# BDD100K original classes (from data_yolo.yaml)
# ['bike', 'bus', 'car', 'drivable area', 'lane', 'motor', 'person', 'rider', 'traffic light', 'traffic sign', 'train', 'truck']
# IDs:  0      1      2        3            4        5        6         7            8              9            10       11

# BDD100K class ID remapping (skip 'lane' which is ID 4)
BDD100K_CLASS_REMAP: Dict[int, Optional[int]] = {
    0: 6,    # bike -> 6
    1: 7,    # bus -> 7
    2: 8,    # car -> 8
    3: 9,    # drivable area -> 9
    4: None, # lane -> SKIP (replaced by Road Lane's specific lane types)
    5: 10,   # motor -> 10
    6: 11,   # person -> 11
    7: 12,   # rider -> 12
    8: 13,   # traffic light -> 13
    9: 14,   # traffic sign -> 14
    10: 15,  # train -> 15
    11: 16,  # truck -> 16
}

# Road Lane classes stay the same (IDs 0-5)
def remap_bdd100k_label(label_path: Path, output_path: Path) -> bool:
    """
    Remap BDD100K label file to unified class IDs.
    
    Returns True if any valid annotations remain after remapping.
    """
    lines_out = []
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            new_class_id = BDD100K_CLASS_REMAP.get(class_id)
            if new_class_id is None:
                continue

            parts[0] = str(new_class_id)
            lines_out.append(' '.join(parts))
    
    if lines_out:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines_out) + '\n')
        return True
    return False

def copy_road_lane_data(source_dir: Path, dest_dir: Path, split: str) -> Dict[str, int]:
    """Copy Road Lane data (no remapping needed)."""
    stats = {'images': 0, 'labels': 0, 'skipped': 0}
    
    src_images = source_dir / split / 'images'
    src_labels = source_dir / split / 'labels'
    dst_images = dest_dir / split / 'images'
    dst_labels = dest_dir / split / 'labels'
    
    if not src_images.exists():
        print(f"  Warning: {src_images} not found")
        return stats
    
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    
    for img_path in src_images.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            new_name = f"roadlane_{img_path.name}"
            shutil.copy2(img_path, dst_images / new_name)
            stats['images'] += 1
            
            label_name = img_path.stem + '.txt'
            label_path = src_labels / label_name
            if label_path.exists():
                new_label_name = f"roadlane_{label_name}"
                shutil.copy2(label_path, dst_labels / new_label_name)
                stats['labels'] += 1
            else:
                stats['skipped'] += 1
    
    return stats

def copy_bdd100k_data(source_dir: Path, dest_dir: Path, split: str) -> Dict[str, int]:
    """Copy BDD100K data with class remapping."""
    stats = {'images': 0, 'labels': 0, 'skipped': 0, 'lane_only': 0}
    
    src_images = source_dir / split / 'images'
    src_labels = source_dir / split / 'labels'
    dst_images = dest_dir / split / 'images'
    dst_labels = dest_dir / split / 'labels'
    
    if not src_images.exists():
        print(f"  Warning: {src_images} not found")
        return stats
    
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    
    for img_path in src_images.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            label_name = img_path.stem + '.txt'
            label_path = src_labels / label_name
            
            if not label_path.exists():
                stats['skipped'] += 1
                continue
            
            new_name = f"bdd100k_{img_path.name}"
            new_label_name = f"bdd100k_{label_name}"
            
            if remap_bdd100k_label(label_path, dst_labels / new_label_name):
                shutil.copy2(img_path, dst_images / new_name)
                stats['images'] += 1
                stats['labels'] += 1
            else:
                stats['lane_only'] += 1
    
    return stats

def validate_dataset(dest_dir: Path) -> Dict:
    """Validate the merged dataset."""
    results = {
        'splits': {},
        'class_counts': defaultdict(int),
        'total_images': 0,
        'total_labels': 0,
        'mismatched': []
    }
    
    for split in ['train', 'valid', 'test']:
        split_dir = dest_dir / split
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not images_dir.exists():
            continue
        
        images = set(p.stem for p in images_dir.glob('*') 
                    if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'])
        labels = set(p.stem for p in labels_dir.glob('*.txt'))
        
        results['splits'][split] = {
            'images': len(images),
            'labels': len(labels),
            'matched': len(images & labels)
        }
        results['total_images'] += len(images)
        results['total_labels'] += len(labels)
        
        # Check class distribution
        for label_path in labels_dir.glob('*.txt'):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        results['class_counts'][class_id] += 1
        
        # Find mismatches
        mismatched = images.symmetric_difference(labels)
        if mismatched:
            results['mismatched'].extend([(split, name) for name in list(mismatched)[:5]])
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Merge Road Lane and BDD100K datasets')
    parser.add_argument('--validate-only', action='store_true', 
                        help='Only validate existing merged dataset')
    parser.add_argument('--data-dir', type=str, 
                        default='data/custom train data',
                        help='Base data directory')
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / args.data_dir
    
    road_lane_dir = data_dir / 'Road Lane.v2i.yolo26'
    bdd100k_dir = data_dir / 'bdd100k'
    unified_dir = data_dir / 'unified'
    
    print("Dataset Merge Configuration")
    print(f"Project root: {project_root}")
    print(f"Road Lane:    {road_lane_dir}")
    print(f"BDD100K:      {bdd100k_dir}")
    print(f"Output:       {unified_dir}")
    print()
    
    if args.validate_only:
        print("Validating existing merged dataset...")
        results = validate_dataset(unified_dir)
        
        print("Validation Results")
        for split, stats in results['splits'].items():
            print(f"\n{split}:")
            print(f"  Images: {stats['images']}")
            print(f"  Labels: {stats['labels']}")
            print(f"  Matched: {stats['matched']}")
        
        print(f"\nTotal Images: {results['total_images']}")
        print(f"Total Labels: {results['total_labels']}")
        
        if results['mismatched']:
            print(f"\nWarning: {len(results['mismatched'])} mismatched files (showing first 5):")
            for split, name in results['mismatched'][:5]:
                print(f"  [{split}] {name}")
        
        print("\nClass Distribution:")
        class_names = [
            'divider-line', 'dotted-line', 'double-line', 'random-line', 
            'road-sign-line', 'solid-line', 'bike', 'bus', 'car', 
            'drivable area', 'motor', 'person', 'rider', 'traffic light', 
            'traffic sign', 'train', 'truck'
        ]
        for class_id in sorted(results['class_counts'].keys()):
            name = class_names[class_id] if class_id < len(class_names) else f"unknown_{class_id}"
            count = results['class_counts'][class_id]
            print(f"  {class_id:2d} ({name}): {count:,}")
        
        return
    
    # Merge datasets
    print("Starting dataset merge...")
    
    for split in ['train', 'valid', 'test']:
        print(f"\nProcessing {split} split...")
        
        # Road Lane
        print("  Copying Road Lane data...")
        rl_stats = copy_road_lane_data(road_lane_dir, unified_dir, split)
        print(f"    Images: {rl_stats['images']}, Labels: {rl_stats['labels']}, Skipped: {rl_stats['skipped']}")
        
        # BDD100K
        print("  Copying BDD100K data (with class remapping)...")
        bdd_stats = copy_bdd100k_data(bdd100k_dir, unified_dir, split)
        print(f"    Images: {bdd_stats['images']}, Labels: {bdd_stats['labels']}, "
              f"Skipped: {bdd_stats['skipped']}, Lane-only: {bdd_stats['lane_only']}")
    
    print("Merge Complete! Validating...")
    results = validate_dataset(unified_dir)
    print(f"\nTotal unified dataset:")
    print(f"  Images: {results['total_images']}")
    print(f"  Labels: {results['total_labels']}")
    
    print("\nClass counts:")
    class_names = [
        'divider-line', 'dotted-line', 'double-line', 'random-line', 
        'road-sign-line', 'solid-line', 'bike', 'bus', 'car', 
        'drivable area', 'motor', 'person', 'rider', 'traffic light', 
        'traffic sign', 'train', 'truck'
    ]
    for class_id in sorted(results['class_counts'].keys()):
        name = class_names[class_id] if class_id < len(class_names) else f"unknown_{class_id}"
        count = results['class_counts'][class_id]
        print(f"  {class_id:2d} ({name}): {count:,}")


if __name__ == '__main__':
    main()