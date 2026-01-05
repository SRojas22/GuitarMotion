"""
Fretboard Data Collection Tool
Interactive tool for capturing varied training images for YOLOv8 fretboard detection
"""

import cv2
import json
import time
from pathlib import Path
from datetime import datetime


class FretboardDataCollector:
    def __init__(self, base_dir="data/raw_images"):
        self.base_dir = Path(base_dir)
        self.session_id = int(time.time())
        self.session_dir = self.base_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Image counter
        self.image_count = 0
        self.target_count = 300

        # Current labels (user can toggle)
        self.lighting = "normal"
        self.angle = "straight"
        self.guitar_type = "acoustic"
        self.background = "plain"

        # Metadata storage
        self.metadata = {
            "session_id": self.session_id,
            "images": []
        }

        # Label options
        self.lighting_options = ["bright", "normal", "dim"]
        self.angle_options = ["straight", "tilt-left", "tilt-right", "top-down"]
        self.guitar_options = ["acoustic", "classical", "electric"]
        self.background_options = ["plain", "cluttered"]

    def cycle_option(self, current, options):
        """Cycle through options list"""
        idx = options.index(current)
        return options[(idx + 1) % len(options)]

    def capture_image(self, frame):
        """Save current frame with metadata"""
        self.image_count += 1
        filename = f"img_{self.image_count:04d}.jpg"
        filepath = self.session_dir / filename

        # Save image
        cv2.imwrite(str(filepath), frame)

        # Record metadata
        self.metadata["images"].append({
            "filename": filename,
            "lighting": self.lighting,
            "angle": self.angle,
            "guitar_type": self.guitar_type,
            "background": self.background,
            "timestamp": datetime.now().isoformat()
        })

        print(f"✅ Captured: {filename} ({self.image_count}/{self.target_count})")
        return True

    def save_metadata(self):
        """Save session metadata to JSON"""
        metadata_path = self.session_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"💾 Metadata saved: {metadata_path}")

    def draw_ui(self, frame):
        """Draw collection UI overlay"""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Semi-transparent background for HUD
        cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Title
        cv2.putText(frame, "Fretboard Data Collection Tool", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Progress bar
        progress = min(self.image_count / self.target_count, 1.0)
        bar_width = 400
        bar_x = 10
        bar_y = 50
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20),
                     (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + int(bar_width * progress), bar_y + 20),
                     (0, 255, 0), -1)
        cv2.putText(frame, f"{self.image_count}/{self.target_count} images",
                   (bar_x + bar_width + 10, bar_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Current settings
        y_offset = 85
        settings = [
            f"[L] Lighting: {self.lighting}",
            f"[A] Angle: {self.angle}",
            f"[G] Guitar: {self.guitar_type}",
            f"[B] Background: {self.background}"
        ]

        for i, setting in enumerate(settings):
            x_pos = 10 + (i % 2) * 250
            y_pos = y_offset + (i // 2) * 25
            cv2.putText(frame, setting, (x_pos, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Instructions at bottom
        instructions = [
            "[SPACE] Capture Image",
            "[Q] Quit & Save"
        ]

        y_bottom = h - 60
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, y_bottom + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Suggestion box (where to position guitar)
        suggested_x = w // 6
        suggested_y = h // 3 + 50
        suggested_w = 2 * w // 3
        suggested_h = h // 3
        cv2.rectangle(frame, (suggested_x, suggested_y),
                     (suggested_x + suggested_w, suggested_y + suggested_h),
                     (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Position fretboard here", (suggested_x + 10, suggested_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return frame

    def run(self):
        """Main collection loop"""
        print("\n" + "="*60)
        print("🎸 FRETBOARD DATA COLLECTION TOOL")
        print("="*60)
        print("This tool helps you collect varied training images.")
        print("\nInstructions:")
        print("  • Position your guitar fretboard in the yellow box")
        print("  • Vary lighting, angles, backgrounds as you capture")
        print("  • Use keyboard shortcuts to toggle current labels")
        print("  • Press SPACE to capture each image")
        print(f"  • Target: {self.target_count} images for good training")
        print("\nKeyboard shortcuts:")
        print("  SPACE - Capture image")
        print("  L - Cycle lighting (bright/normal/dim)")
        print("  A - Cycle angle (straight/tilt/top-down)")
        print("  G - Cycle guitar type (acoustic/classical/electric)")
        print("  B - Cycle background (plain/cluttered)")
        print("  Q - Quit and save session")
        print("="*60 + "\n")

        web_cam = cv2.VideoCapture(0)

        if not web_cam.isOpened():
            print("❌ Failed to open camera")
            return

        print(f"📁 Session directory: {self.session_dir}")
        print("📸 Camera ready! Start capturing...\n")

        while web_cam.isOpened():
            success, frame = web_cam.read()
            if not success:
                print("❌ Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)

            # Draw UI
            display_frame = self.draw_ui(frame)

            cv2.imshow("Fretboard Data Collection", display_frame)

            key = cv2.waitKey(1) & 0xFF

            # Capture image
            if key == ord(' '):
                self.capture_image(frame)

            # Toggle lighting
            elif key == ord('l'):
                self.lighting = self.cycle_option(self.lighting, self.lighting_options)
                print(f"Lighting → {self.lighting}")

            # Toggle angle
            elif key == ord('a'):
                self.angle = self.cycle_option(self.angle, self.angle_options)
                print(f"Angle → {self.angle}")

            # Toggle guitar type
            elif key == ord('g'):
                self.guitar_type = self.cycle_option(self.guitar_type, self.guitar_options)
                print(f"Guitar type → {self.guitar_type}")

            # Toggle background
            elif key == ord('b'):
                self.background = self.cycle_option(self.background, self.background_options)
                print(f"Background → {self.background}")

            # Quit
            elif key == ord('q'):
                print("\n📷 Stopping collection...")
                break

        web_cam.release()
        cv2.destroyAllWindows()

        # Save metadata
        self.save_metadata()

        # Session summary
        print("\n" + "="*60)
        print("📊 SESSION SUMMARY")
        print("="*60)
        print(f"Images captured: {self.image_count}")
        print(f"Session directory: {self.session_dir}")
        print(f"Metadata saved: {self.session_dir / 'metadata.json'}")

        # Variety breakdown
        lighting_counts = {}
        angle_counts = {}
        guitar_counts = {}
        background_counts = {}

        for img in self.metadata["images"]:
            lighting_counts[img["lighting"]] = lighting_counts.get(img["lighting"], 0) + 1
            angle_counts[img["angle"]] = angle_counts.get(img["angle"], 0) + 1
            guitar_counts[img["guitar_type"]] = guitar_counts.get(img["guitar_type"], 0) + 1
            background_counts[img["background"]] = background_counts.get(img["background"], 0) + 1

        print("\nVariety breakdown:")
        print(f"  Lighting: {lighting_counts}")
        print(f"  Angles: {angle_counts}")
        print(f"  Guitar types: {guitar_counts}")
        print(f"  Backgrounds: {background_counts}")

        if self.image_count < self.target_count:
            print(f"\n⚠️  Recommendation: Capture {self.target_count - self.image_count} more images")
            print("   for better model performance.")
        else:
            print(f"\n✅ Great! You've reached the target of {self.target_count} images.")

        print("\n📝 Next steps:")
        print("  1. Label images with LabelImg (see docs/LABELING_GUIDE.md)")
        print("  2. Organize labeled data with tools/organize_labeled_data.py")
        print("  3. Train model with models/train_fretboard_detector.py")
        print("="*60 + "\n")


if __name__ == "__main__":
    collector = FretboardDataCollector()
    collector.run()
