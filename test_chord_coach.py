"""
Test Chord Coach Mode
Complete test of chord coach functionality
"""

import cv2
from vision.hybrid_detector import HybridDetector
from vision.lock_state import LockStateManager
from vision.fretboard_mapper import FretboardMapper
from ui.overlays import (
    draw_lock_state_ui,
    draw_fretboard_glow,
    draw_progress_bar,
    draw_instructions
)
from ui.calibration_ui import two_click_calibration, show_calibration_preview
from modes.chord_coach_session import run_chord_coach


def test_chord_coach():
    """Test complete chord coach workflow"""
    print("\n" + "="*60)
    print("🎸 CHORD COACH TEST")
    print("="*60)
    print("This will test the complete chord coach system:")
    print("  1. Detect and lock fretboard")
    print("  2. Calibrate with two clicks")
    print("  3. Start chord practice session")
    print("="*60 + "\n")

    # Phase 1: Detect and lock
    print("PHASE 1: Detection & Lock")
    detector = HybridDetector()
    lock_manager = LockStateManager(lock_threshold=0.75, stable_frames=15)

    web_cam = cv2.VideoCapture(0)
    last_bbox = None
    calibration_frame = None

    if not web_cam.isOpened():
        print("❌ Failed to open camera")
        return False

    print("📸 Camera opened - position your guitar...")
    print("   Press SPACE when locked")
    print("   Press Q to quit\n")

    while web_cam.isOpened():
        success, frame = web_cam.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        # Detect
        bbox, conf, method = detector.detect(frame)
        state = lock_manager.update(bbox, conf)
        progress = lock_manager.get_lock_progress()

        # Draw UI
        frame = draw_lock_state_ui(frame, state, conf, method)

        if bbox is not None:
            frame = draw_fretboard_glow(frame, bbox)
            last_bbox = bbox

        if state.name == 'LOCKING':
            frame = draw_progress_bar(frame, progress, position='bottom')

        frame = draw_instructions(frame, locked=lock_manager.is_locked())

        cv2.imshow("Chord Coach Test - Lock", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and lock_manager.is_locked():
            # Adjust bbox position - move it up slightly for better framing
            VERTICAL_OFFSET = 40  # pixels to move up
            x1, y1, x2, y2 = last_bbox
            y1_adjusted = max(0, y1 - VERTICAL_OFFSET)
            y2_adjusted = max(0, y2 - VERTICAL_OFFSET)
            last_bbox = [x1, y1_adjusted, x2, y2_adjusted]

            calibration_frame = cv2.flip(web_cam.read()[1], 1)
            break
        elif key == ord('q'):
            print("❌ Test cancelled")
            web_cam.release()
            cv2.destroyAllWindows()
            return False

    web_cam.release()
    cv2.destroyAllWindows()

    if last_bbox is None or calibration_frame is None:
        print("❌ Failed to detect fretboard")
        return False

    print(f"✅ Fretboard locked!\n")

    # Phase 2: Calibration
    print("PHASE 2: Two-Click Calibration")
    nut_x, fret_12_x = two_click_calibration(calibration_frame, last_bbox)

    if nut_x is None or fret_12_x is None:
        print("❌ Calibration failed")
        return False

    # Create mapper
    mapper = FretboardMapper(last_bbox, num_strings=6, num_frets=20)
    mapper.set_reference_points(nut_x, fret_12_x)

    # Preview
    print("\nPHASE 3: Calibration Preview")
    show_calibration_preview(calibration_frame, mapper, duration_ms=2000)

    # Phase 4: Chord Coach Session
    print("\nPHASE 4: Chord Coach Session")
    print("="*60)
    run_chord_coach(mapper, detector)

    print("\n" + "="*60)
    print("✅ CHORD COACH TEST COMPLETE!")
    print("="*60 + "\n")

    return True


if __name__ == "__main__":
    print("\n🎸 GuitarMotion - Chord Coach Test\n")

    success = test_chord_coach()

    if success:
        print("🎉 All systems working!")
        print("\n📝 Chord Coach is ready to use in main app!")
        print("   Run: python main.py")
        print("   Select: Mode 2 (Chord Coach)")
    else:
        print("⚠️  Test incomplete or failed")
        print("\n💡 Troubleshooting:")
        print("   • Check guitar is visible and well-lit")
        print("   • Make sure venv is activated")
        print("   • Verify all dependencies are installed")
