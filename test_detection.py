"""
Test Fretboard Detection
Quick test to verify your trained model works with live camera
"""

import cv2
from vision.hybrid_detector import HybridDetector
from vision.lock_state import LockStateManager
from ui.overlays import (
    draw_lock_state_ui,
    draw_fretboard_glow,
    draw_progress_bar,
    draw_instructions,
    draw_detection_stats
)


def test_detection():
    """Test live fretboard detection with your trained model"""
    print("\n" + "="*60)
    print("FRETBOARD DETECTION TEST")
    print("="*60)
    print("This will test your trained model with live camera feed.")
    print("\nControls:")
    print("  SPACE - Confirm lock and exit")
    print("  Q - Quit")
    print("  S - Toggle stats display")
    print("="*60 + "\n")

    # Initialize detector and lock manager
    detector = HybridDetector()
    lock_manager = LockStateManager(lock_threshold=0.75, stable_frames=15)

    # Open camera
    web_cam = cv2.VideoCapture(0)
    show_stats = False
    last_bbox = None

    if not web_cam.isOpened():
        print("Failed to open camera")
        return False

    print("Camera opened successfully!")
    print("Position your guitar in frame...\n")

    while web_cam.isOpened():
        success, frame = web_cam.read()
        if not success:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)

        # Detect fretboard
        bbox, conf, method = detector.detect(frame)

        # Update lock state
        state = lock_manager.update(bbox, conf)
        progress = lock_manager.get_lock_progress()

        # Draw UI overlays
        frame = draw_lock_state_ui(frame, state, conf, method)

        if bbox is not None:
            # Draw glow around detected fretboard
            frame = draw_fretboard_glow(frame, bbox)
            last_bbox = bbox

            # Optional stats display
            if show_stats:
                frame = draw_detection_stats(frame, bbox, conf, method)

        # Draw progress bar when locking
        if state.name == 'LOCKING':
            frame = draw_progress_bar(frame, progress, position='bottom')

        # Draw instructions
        frame = draw_instructions(frame, locked=lock_manager.is_locked())

        # Show frame
        cv2.imshow("GuitarMotion - Detection Test", frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and lock_manager.is_locked():
            print(f"\nLock confirmed!")
            print(f"   Method: {method.upper()}")
            print(f"   Confidence: {conf:.1%}")
            print(f"   Bbox: {bbox}")
            break
        elif key == ord('q'):
            print("\nTest cancelled")
            break
        elif key == ord('s'):
            show_stats = not show_stats
            print(f"Stats display: {'ON' if show_stats else 'OFF'}")

    web_cam.release()
    cv2.destroyAllWindows()

    if lock_manager.is_locked():
        print("\nDetection test successful!")
        print("Your trained model is working perfectly!")
        return True
    else:
        print("\nTest ended without achieving lock")
        return False

