"""
Chord Coach Session
Interactive chord practice mode with real-time feedback
"""

import cv2
import mediapipe as mp
import time
from .chord_coach import ChordCoach


def run_chord_coach(mapper, detector):
    """
    Main chord coach practice session

    Args:
        mapper: Calibrated FretboardMapper instance
        detector: HybridDetector instance (for continuous tracking)

    Returns:
        bool: True if session completed successfully
    """
    print("\n" + "="*60)
    print("🎸 CHORD COACH MODE")
    print("="*60)
    print("Learn and practice guitar chords with real-time feedback!")
    print("="*60 + "\n")

    # Initialize chord coach
    coach = ChordCoach()
    coach.set_mapper(mapper)

    # Show available chords
    available_chords = coach.get_available_chords()
    print("Available chords:")
    print("  Beginner: C, G, D, Em, Am")
    print("  Intermediate: E, A, Dm, F")
    print("  Advanced: Cadd9, Dsus4, Em7, A7sus4")
    print()

    # Let user select chord
    while True:
        chord_name = input("Enter chord to practice (or 'q' to quit): ").strip()

        if chord_name.lower() == 'q':
            print("Exiting chord coach mode")
            return False

        if coach.select_chord(chord_name):
            break
        else:
            print(f"⚠️  '{chord_name}' not found. Try: {', '.join(available_chords[:5])}")

    print(f"\n✅ Starting practice session for: {coach.current_chord['name']}")
    print("="*60)
    print("Instructions:")
    print("  • Position your fingers on the fretboard")
    print("  • Ghost circles show where to place fingers")
    print("  • Green = correct, Red = wrong position")
    print("  • Press 'c' to change chord")
    print("  • Press 'q' to quit")
    print("="*60 + "\n")

    # Setup hand tracking
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # Setup camera
    web_cam = cv2.VideoCapture(0)

    if not web_cam.isOpened():
        print("❌ Failed to open camera")
        return False

    # Session stats
    session_start = time.time()
    perfect_count = 0
    frame_count = 0
    accuracy_history = []

    print("📸 Camera active - start practicing!\n")

    while web_cam.isOpened():
        success, frame = web_cam.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_count += 1

        # Draw fretboard overlay
        frame = mapper.draw_overlay(frame, alpha=0.3)

        # Draw target chord overlay (ghost circles)
        frame = coach.draw_target_overlay(frame, color=(100, 150, 255), alpha=0.6)

        # Detect hands
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        score = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 200), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2)
                )

                # Check finger placement
                detected_fingers = coach.check_finger_placement(
                    hand_landmarks, (h, w, 3)
                )

                # Score placement
                score = coach.score_placement(detected_fingers)

                # Draw detected fingers with color coding
                if score:
                    frame = coach.draw_detected_fingers(frame, detected_fingers, score)

        # Draw feedback overlay
        if score:
            frame = coach.draw_feedback(frame, score)
            accuracy_history.append(score['accuracy'])

            # Track perfect hits
            if score['accuracy'] >= 0.95:
                perfect_count += 1

        # Draw chord name
        frame = coach.draw_chord_diagram(frame, position='top-right')

        # Draw session stats
        elapsed = int(time.time() - session_start)
        stats_y = h - 60
        cv2.putText(frame, f"Time: {elapsed}s", (10, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Perfect: {perfect_count}", (10, stats_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Average accuracy (last 30 frames)
        if len(accuracy_history) > 0:
            recent_avg = sum(accuracy_history[-30:]) / min(len(accuracy_history), 30)
            cv2.putText(frame, f"Avg: {recent_avg:.0%}", (10, stats_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)

        # Show frame
        cv2.imshow("Chord Coach", frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n✅ Practice session ended")
            break
        elif key == ord('c'):
            # Change chord
            web_cam.release()
            cv2.destroyAllWindows()
            print("\n" + "="*60)
            print("Change Chord")
            print("="*60)

            new_chord = input("Enter new chord name: ").strip()
            if coach.select_chord(new_chord):
                # Restart session
                perfect_count = 0
                accuracy_history.clear()
                session_start = time.time()
                web_cam = cv2.VideoCapture(0)
            else:
                print(f"⚠️  Chord not found. Continuing with {coach.current_chord_name}")
                web_cam = cv2.VideoCapture(0)

    web_cam.release()
    cv2.destroyAllWindows()

    # Print session summary
    print("\n" + "="*60)
    print("📊 SESSION SUMMARY")
    print("="*60)
    print(f"Chord practiced: {coach.current_chord['name']}")
    print(f"Duration: {int(time.time() - session_start)}s")
    print(f"Perfect frames: {perfect_count}")

    if len(accuracy_history) > 0:
        avg_accuracy = sum(accuracy_history) / len(accuracy_history)
        print(f"Average accuracy: {avg_accuracy:.1%}")
    else:
        print("No accuracy data recorded")

    print("="*60 + "\n")

    return True


def quick_chord_test(mapper):
    """
    Quick test to verify chord detection works

    Args:
        mapper: Calibrated FretboardMapper

    Returns:
        bool: True if test successful
    """
    print("\n🧪 Quick Chord Coach Test")
    coach = ChordCoach()
    coach.set_mapper(mapper)

    # Select a simple chord
    if not coach.select_chord("Em"):
        print("❌ Test failed - couldn't load Em chord")
        return False

    print("✅ Chord coach initialized successfully!")
    print(f"   Target: {coach.current_chord['name']}")
    print(f"   Fingers needed: {len(coach.current_chord['fingers'])}")

    return True
