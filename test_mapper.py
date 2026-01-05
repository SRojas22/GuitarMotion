"""
Test Fretboard Mapper
Test the two-click calibration and string/fret overlay visualization
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
from ui.calibration_ui import (
    two_click_calibration,
    show_calibration_preview,
    interactive_calibration_test
)


def test_mapper():
    """Test the complete mapper workflow: detect → lock → calibrate → visualize"""
    print("\n" + "="*60)
    print("FRETBOARD MAPPER TEST")
    print("="*60)
    print("This will test the complete mapping workflow:")
    print("  1. Detect and lock onto fretboard")
    print("  2. Two-click calibration (nut + 12th fret)")
    print("  3. Visualize string/fret overlay")
    print("  4. Interactive test mode")
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
    print("   Press SPACE when locked to continue")
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

        cv2.imshow("Mapper Test - Lock", frame)

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

    print(f"✅ Fretboard locked!")
    print(f"   Bbox: {last_bbox}\n")

    # Phase 2: Two-click calibration
    print("PHASE 2: Two-Click Calibration")
    nut_x, fret_12_x = two_click_calibration(calibration_frame, last_bbox)

    if nut_x is None or fret_12_x is None:
        print("❌ Calibration failed")
        return False

    # Create and calibrate mapper (20 frets for full range)
    mapper = FretboardMapper(last_bbox, num_strings=6, num_frets=20)
    mapper.set_reference_points(nut_x, fret_12_x)

    # Phase 3: Show preview
    print("\nPHASE 3: Preview")
    show_calibration_preview(calibration_frame, mapper, duration_ms=3000)

    # Phase 4: Interactive test
    print("\nPHASE 4: Interactive Test")
    interactive_calibration_test(calibration_frame, mapper)

    print("\n" + "="*60)
    print("✅ MAPPER TEST COMPLETE!")
    print("="*60)
    print("Mapper details:")
    print(f"  • Strings: {mapper.num_strings}")
    print(f"  • Frets: {mapper.num_frets}")
    print(f"  • Nut position: {mapper.nut_x}")
    print(f"  • 12th fret: {mapper.fret_12_x}")
    print(f"  • Scale length: {(mapper.fret_12_x - mapper.nut_x) * 2}px")
    print("="*60 + "\n")

    return True


if __name__ == "__main__":
    success = test_mapper()

    if success:
        print("🎉 All tests passed!")
        print("\n📝 The mapper is ready for:")
        print("   • Chord Coach mode (Phase C)")
        print("   • Play-Along mode (Phase D)")
        print("   • AR overlays in the main app")
    else:
        print("⚠️  Test incomplete or failed")
        print("\n💡 Tips:")
        print("   • Make sure your guitar fretboard is clearly visible")
        print("   • Use good lighting")
        print("   • Click precisely on the nut and 12th fret")
