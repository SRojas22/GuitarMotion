"""
Two-Click Calibration UI
Interactive interface for calibrating fretboard string/fret positions
"""

import cv2
import numpy as np


def two_click_calibration(frame, bbox):
    """
    Interactive two-click calibration for fretboard mapping
    User clicks on: 1) Nut (0th fret), 2) 12th fret

    Args:
        frame: OpenCV frame showing the locked fretboard
        bbox: [x1, y1, x2, y2] bounding box of fretboard

    Returns:
        (nut_x, fret_12_x): X coordinates of nut and 12th fret, or (None, None) if cancelled
    """
    clicks = []
    x1, y1, x2, y2 = map(int, bbox)

    def mouse_callback(event, x, y, flags, param):
        """Handle mouse clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Only accept clicks within bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                clicks.append(x)
                print(f"✓ Click {len(clicks)}: x={x}")
            else:
                print(f"⚠️  Click outside fretboard area - try again")

    window_name = "Calibrate Fretboard"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n" + "="*60)
    print("🎯 FRETBOARD CALIBRATION")
    print("="*60)
    print("Click on two reference points:")
    print("  1. The NUT (leftmost edge / 0th fret)")
    print("  2. The 12TH FRET")
    print("\nPress 'q' to cancel")
    print("="*60 + "\n")

    while len(clicks) < 2:
        display = frame.copy()

        # Draw bbox outline
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Instructions overlay
        if len(clicks) == 0:
            # First click - nut
            instruction = "STEP 1: Click on the NUT (leftmost edge)"
            color = (0, 255, 255)  # Yellow

            # Draw arrow pointing to left edge
            arrow_start = (x1 - 30, (y1 + y2) // 2)
            arrow_end = (x1 + 10, (y1 + y2) // 2)
            cv2.arrowedLine(display, arrow_start, arrow_end, color, 3, tipLength=0.3)

        elif len(clicks) == 1:
            # Second click - 12th fret
            instruction = "STEP 2: Click on the 12TH FRET"
            color = (0, 255, 255)  # Yellow

            # Draw first click as vertical line
            cv2.line(display, (clicks[0], y1), (clicks[0], y2), (0, 255, 0), 3)
            cv2.putText(display, "NUT", (clicks[0] - 20, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw guide for 12th fret (roughly in middle-right area)
            estimated_12th = x1 + int((x2 - x1) * 0.6)
            cv2.line(display, (estimated_12th, y1), (estimated_12th, y2),
                    (100, 100, 100), 1, cv2.LINE_AA)
            cv2.putText(display, "12th? (approx)", (estimated_12th - 50, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Draw instruction text at top
        overlay = display.copy()
        text_bg_y1 = 10
        text_bg_y2 = 50
        cv2.rectangle(overlay, (10, text_bg_y1), (x2, text_bg_y2), (0, 0, 0), -1)
        display = cv2.addWeighted(overlay, 0.7, display, 0.3, 0)

        cv2.putText(display, instruction, (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw click counter
        progress = f"Clicks: {len(clicks)}/2"
        cv2.putText(display, progress, (x2 - 120, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("❌ Calibration cancelled")
            cv2.destroyWindow(window_name)
            return None, None

    cv2.destroyWindow(window_name)

    nut_x = clicks[0]
    fret_12_x = clicks[1]

    # Validate: 12th fret should be to the right of nut
    if fret_12_x <= nut_x:
        print("⚠️  Error: 12th fret must be to the right of the nut!")
        print("   Please ensure clicks are: 1) Nut (left), 2) 12th fret (right)")
        return None, None

    print(f"\n✅ Calibration complete!")
    print(f"   Nut X: {nut_x}")
    print(f"   12th Fret X: {fret_12_x}")
    print(f"   Scale length: {(fret_12_x - nut_x) * 2}px")

    return nut_x, fret_12_x


def show_calibration_preview(frame, mapper, duration_ms=2000):
    """
    Show preview of calibrated fretboard with overlay

    Args:
        frame: OpenCV frame
        mapper: FretboardMapper instance (calibrated)
        duration_ms: How long to show preview (default 2 seconds)
    """
    if not mapper.is_calibrated:
        return

    print("\n📸 Showing calibration preview...")

    window_name = "Calibration Preview"
    preview = mapper.draw_overlay(frame.copy(), alpha=0.6)

    # Add success banner
    h, w = preview.shape[:2]
    overlay = preview.copy()
    cv2.rectangle(overlay, (w//4, h//2 - 40), (3*w//4, h//2 + 40), (0, 255, 0), -1)
    preview = cv2.addWeighted(overlay, 0.3, preview, 0.7, 0)

    cv2.putText(preview, "Calibration Successful!", (w//4 + 20, h//2),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(preview, "Press any key to continue", (w//4 + 40, h//2 + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow(window_name, preview)
    cv2.waitKey(duration_ms)
    cv2.destroyWindow(window_name)


def interactive_calibration_test(frame, mapper):
    """
    Interactive test mode - click anywhere to see string/fret detection

    Args:
        frame: OpenCV frame
        mapper: Calibrated FretboardMapper

    Returns:
        bool: True if test completed, False if cancelled
    """
    if not mapper.is_calibrated:
        print("⚠️  Mapper not calibrated - cannot run test")
        return False

    current_pos = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_pos
        if event == cv2.EVENT_MOUSEMOVE:
            current_pos = (x, y)

    window_name = "Calibration Test"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n" + "="*60)
    print("🧪 CALIBRATION TEST MODE")
    print("="*60)
    print("Move your mouse over the fretboard to test detection")
    print("Press 'q' to finish test")
    print("="*60 + "\n")

    while True:
        display = mapper.draw_overlay(frame.copy(), alpha=0.6)

        if current_pos:
            x, y = current_pos
            string_num, fret_num = mapper.get_string_fret_from_point(x, y)

            if string_num is not None and fret_num is not None:
                # Draw cursor position
                cv2.circle(display, (x, y), 8, (0, 255, 255), 2)
                cv2.circle(display, (x, y), 3, (0, 255, 255), -1)

                # Highlight detected position
                display = mapper.draw_position_highlight(display, string_num, fret_num)

                # Show info
                info = f"String {string_num + 1} | Fret {fret_num}"
                cv2.putText(display, info, (x + 15, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                # Out of range
                cv2.circle(display, (x, y), 8, (0, 0, 255), 2)

        # Instructions
        cv2.putText(display, "Move mouse to test - Press Q to exit",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyWindow(window_name)
    print("✅ Test complete!")
    return True
