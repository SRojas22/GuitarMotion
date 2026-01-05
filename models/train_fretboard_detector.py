"""
YOLOv8 Fretboard Detector Training Script
Trains a custom YOLOv8 model to detect guitar fretboards
"""

from ultralytics import YOLO
from pathlib import Path
import argparse


def train_model(
    data_yaml='data/labeled/data.yaml',
    model_size='n',  # n (nano), s (small), m (medium), l (large), x (xlarge)
    epochs=100,
    img_size=640,
    batch_size=16,
    patience=20,
    name='fretboard_detector'
):
    """
    Train YOLOv8 fretboard detector

    Args:
        data_yaml: Path to data.yaml configuration file
        model_size: YOLOv8 model size (n, s, m, l, x)
        epochs: Maximum number of training epochs
        img_size: Input image size
        batch_size: Batch size for training
        patience: Early stopping patience (epochs without improvement)
        name: Name for this training run
    """

    print("\n" + "="*60)
    print("🎸 YOLOV8 FRETBOARD DETECTOR TRAINING")
    print("="*60)

    # Verify data.yaml exists
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"\n❌ Error: data.yaml not found at {data_path}")
        print("   Run tools/organize_labeled_data.py first to create it.")
        return

    print(f"\n📁 Dataset config: {data_path}")
    print(f"🤖 Model size: YOLOv8{model_size}")
    print(f"📊 Training params:")
    print(f"   - Epochs: {epochs}")
    print(f"   - Image size: {img_size}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Early stopping patience: {patience}")

    # Load pretrained YOLOv8 model
    model_name = f'yolov8{model_size}.pt'
    print(f"\n⬇️  Loading pretrained model: {model_name}")

    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   Make sure ultralytics is installed: pip install ultralytics")
        return

    print("✅ Model loaded successfully")

    # Train
    print(f"\n🚀 Starting training (this may take a while)...\n")

    try:
        results = model.train(
            data=str(data_path),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name=name,
            patience=patience,
            save=True,
            plots=True,
            verbose=True,
            device='cpu'  # Use 'cuda' or 'mps' if GPU available
        )

        print("\n✅ Training complete!")

        # Validate the model
        print("\n📊 Running validation...")
        metrics = model.val()

        print("\n" + "="*60)
        print("📈 TRAINING RESULTS")
        print("="*60)
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.p[0]:.4f}")
        print(f"Recall: {metrics.box.r[0]:.4f}")

        # Interpret results
        print("\n📊 Performance Assessment:")
        if metrics.box.map50 >= 0.90:
            print("   🌟 Excellent! (mAP50 >= 0.90)")
        elif metrics.box.map50 >= 0.85:
            print("   ✅ Good! (mAP50 >= 0.85)")
        elif metrics.box.map50 >= 0.75:
            print("   ⚠️  Acceptable (mAP50 >= 0.75)")
            print("      Consider collecting more varied images")
        else:
            print("   ❌ Needs improvement (mAP50 < 0.75)")
            print("      Recommendations:")
            print("      - Collect more training images (aim for 500+)")
            print("      - Ensure good variety in lighting/angles")
            print("      - Relabel images for tighter bounding boxes")
            print("      - Try a larger model (yolov8s or yolov8m)")

        # Find best weights
        runs_dir = Path('runs/detect') / name
        best_weights = runs_dir / 'weights' / 'best.pt'

        if best_weights.exists():
            # Copy to models/weights/
            dest_path = Path('models/weights/fretboard_detector.pt')
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            import shutil
            shutil.copy2(best_weights, dest_path)

            print(f"\n💾 Best model saved to: {dest_path}")
            print(f"   (Original: {best_weights})")
        else:
            print(f"\n⚠️  Warning: Best weights not found at {best_weights}")

        print(f"\n📁 Full results saved to: {runs_dir}")
        print(f"   - Training plots: {runs_dir}")
        print(f"   - Validation results: {runs_dir}")
        print(f"   - Model weights: {runs_dir / 'weights'}")

        print("\n📝 Next steps:")
        print("  1. Review training plots in the runs directory")
        print("  2. If mAP50 < 0.85, consider:")
        print("     - Collecting more images")
        print("     - Improving label quality")
        print("     - Training for more epochs")
        print("  3. If satisfied, proceed to Phase A:")
        print("     - Implement fretboard detection in main app")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 fretboard detector')

    parser.add_argument('--data', default='data/labeled/data.yaml',
                       help='Path to data.yaml')
    parser.add_argument('--model', default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size: n(ano), s(mall), m(edium), l(arge), x(large)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--name', default='fretboard_detector',
                       help='Name for training run')

    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch,
        patience=args.patience,
        name=args.name
    )


if __name__ == "__main__":
    main()
