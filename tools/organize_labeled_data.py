"""
Labeled Data Organization Tool
Organizes LabelImg output into train/val/test split for YOLOv8 training
"""

import shutil
import random
from pathlib import Path
import yaml


class DataOrganizer:
    def __init__(self, source_dir="data/raw_images", output_dir="data/labeled"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)

        # Create directory structure
        self.image_dirs = {
            'train': self.output_dir / 'images' / 'train',
            'val': self.output_dir / 'images' / 'val',
            'test': self.output_dir / 'images' / 'test'
        }

        self.label_dirs = {
            'train': self.output_dir / 'labels' / 'train',
            'val': self.output_dir / 'labels' / 'val',
            'test': self.output_dir / 'labels' / 'test'
        }

        # Create all directories
        for dirs in [self.image_dirs, self.label_dirs]:
            for dir_path in dirs.values():
                dir_path.mkdir(parents=True, exist_ok=True)

    def find_labeled_pairs(self):
        """Find all image+label pairs from session folders"""
        pairs = []

        # Search all session folders
        for session_dir in self.source_dir.glob("session_*"):
            if not session_dir.is_dir():
                continue

            # Find all images
            for img_path in session_dir.glob("*.jpg"):
                # Check if corresponding label exists
                label_path = img_path.with_suffix('.txt')

                if label_path.exists():
                    pairs.append({
                        'image': img_path,
                        'label': label_path
                    })

        return pairs

    def split_dataset(self, pairs, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """Split pairs into train/val/test"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, \
            "Ratios must sum to 1.0"

        # Shuffle for random split
        random.shuffle(pairs)

        total = len(pairs)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        splits = {
            'train': pairs[:train_end],
            'val': pairs[train_end:val_end],
            'test': pairs[val_end:]
        }

        return splits

    def copy_files(self, splits):
        """Copy images and labels to organized structure"""
        stats = {'train': 0, 'val': 0, 'test': 0}

        for split_name, pairs in splits.items():
            print(f"\n📁 Copying {split_name} set...")

            for pair in pairs:
                # Copy image
                img_dest = self.image_dirs[split_name] / pair['image'].name
                shutil.copy2(pair['image'], img_dest)

                # Copy label
                label_dest = self.label_dirs[split_name] / pair['label'].name
                shutil.copy2(pair['label'], label_dest)

                stats[split_name] += 1

            print(f"   ✅ Copied {stats[split_name]} pairs")

        return stats

    def create_data_yaml(self):
        """Create data.yaml for YOLOv8 training"""
        data_yaml = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,
            'names': ['fretboard']
        }

        yaml_path = self.output_dir / 'data.yaml'

        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"\n📝 Created data.yaml: {yaml_path}")
        return yaml_path

    def run(self):
        """Main organization workflow"""
        print("\n" + "="*60)
        print("🗂️  LABELED DATA ORGANIZATION TOOL")
        print("="*60)
        print("This tool organizes your labeled images for YOLOv8 training.")
        print("\nSearching for labeled image pairs...")

        # Find all labeled pairs
        pairs = self.find_labeled_pairs()

        if not pairs:
            print("\n❌ No labeled pairs found!")
            print("   Make sure you've labeled images with LabelImg and")
            print("   saved the .txt files in the same directory as images.")
            print(f"   Searched in: {self.source_dir}")
            return

        print(f"✅ Found {len(pairs)} labeled image pairs")

        # Split dataset
        print("\n📊 Splitting dataset (80% train, 10% val, 10% test)...")
        splits = self.split_dataset(pairs)

        print(f"   Train: {len(splits['train'])} images")
        print(f"   Val: {len(splits['val'])} images")
        print(f"   Test: {len(splits['test'])} images")

        # Confirm before copying
        response = input("\n⚠️  This will copy files to data/labeled/. Continue? (y/n): ")
        if response.lower() != 'y':
            print("❌ Cancelled.")
            return

        # Copy files
        stats = self.copy_files(splits)

        # Create data.yaml
        yaml_path = self.create_data_yaml()

        # Summary
        print("\n" + "="*60)
        print("✅ ORGANIZATION COMPLETE")
        print("="*60)
        print(f"Output directory: {self.output_dir}")
        print(f"\nDataset split:")
        print(f"  Train: {stats['train']} images")
        print(f"  Val: {stats['val']} images")
        print(f"  Test: {stats['test']} images")
        print(f"  Total: {sum(stats.values())} images")

        print(f"\n📁 Directory structure:")
        print(f"  {self.output_dir}/")
        print(f"    ├── images/")
        print(f"    │   ├── train/ ({stats['train']} images)")
        print(f"    │   ├── val/ ({stats['val']} images)")
        print(f"    │   └── test/ ({stats['test']} images)")
        print(f"    ├── labels/")
        print(f"    │   ├── train/ ({stats['train']} labels)")
        print(f"    │   ├── val/ ({stats['val']} labels)")
        print(f"    │   └── test/ ({stats['test']} labels)")
        print(f"    └── data.yaml")

        print("\n📝 Next steps:")
        print("  1. Verify the split looks correct")
        print(f"  2. Review data.yaml: {yaml_path}")
        print("  3. Train the model:")
        print("     python models/train_fretboard_detector.py")
        print("="*60 + "\n")


def main():
    # Check if source directory exists
    source_dir = Path("data/raw_images")

    if not source_dir.exists():
        print(f"❌ Error: Source directory not found: {source_dir}")
        print("   Run tools/collect_fretboard_data.py first to collect images.")
        return

    organizer = DataOrganizer()
    organizer.run()


if __name__ == "__main__":
    main()
