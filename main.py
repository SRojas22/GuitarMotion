import cv2
import mediapipe as mp
import sounddevice as sd
from scipy.io.wavfile import write
import threading
import numpy as np
import librosa as lb
import json
import time
import os
from pathlib import Path
from src.guitar_detection import GuitarDetector


stop_event = threading.Event()
hand_positions = []  # Store hand positions with timestamps

# Guitar standard tuning notes (from low E to high E)
GUITAR_STRINGS = {
    0: "E2",  # 6th string (lowest)
    1: "A2",  # 5th string
    2: "D3",  # 4th string
    3: "G3",  # 3rd string
    4: "B3",  # 2nd string
    5: "E4"   # 1st string (highest)
}

# Fret to semitone mapping (each fret is one semitone)
FRET_NOTES = {
    "E2": ["E2", "F2", "F♯2", "G2", "G♯2", "A2", "A♯2", "B2", "C3", "C♯3", "D3", "D♯3", "E3"],
    "A2": ["A2", "A♯2", "B2", "C3", "C♯3", "D3", "D♯3", "E3", "F3", "F♯3", "G3", "G♯3", "A3"],
    "D3": ["D3", "D♯3", "E3", "F3", "F♯3", "G3", "G♯3", "A3", "A♯3", "B3", "C4", "C♯4", "D4"],
    "G3": ["G3", "G♯3", "A3", "A♯3", "B3", "C4", "C♯4", "D4", "D♯4", "E4", "F4", "F♯4", "G4"],
    "B3": ["B3", "C4", "C♯4", "D4", "D♯4", "E4", "F4", "F♯4", "G4", "G♯4", "A4", "A♯4", "B4"],
    "E4": ["E4", "F4", "F♯4", "G4", "G♯4", "A4", "A♯4", "B4", "C5", "C♯5", "D5", "D♯5", "E5"]
}


def calibrate_guitar():
    """
    Calibration phase: Detect guitar and wait for user confirmation.
    Returns: GuitarDetector object if successful, None otherwise
    """
    print("\n" + "="*60)
    print("🎸 GUITAR CALIBRATION")
    print("="*60)
    print("Position your guitar fretboard so it's:")
    print("  • Horizontal in the frame")
    print("  • Well-lit and clearly visible")
    print("  • Taking up about 1/3 to 1/2 of the frame")
    print("\nPress SPACE when guitar is detected to start session")
    print("Press 'q' to quit")
    print("="*60 + "\n")

    web_cam = cv2.VideoCapture(0)
    detector = GuitarDetector()
    guitar_detected = False
    detected_region = None

    while web_cam.isOpened():
        success, frame = web_cam.read()
        if not success:
            print("Failed to access camera")
            break

        frame = cv2.flip(frame, 1)

        # Attempt to detect guitar
        detected_region = detector.detect_guitar(frame)

        # Draw detection hints and overlay
        frame = detector.draw_detection_hint(frame, detected_region)

        cv2.imshow("Guitar Calibration", frame)

        key = cv2.waitKey(1) & 0xFF

        # Press SPACE to confirm guitar detection
        if key == ord(' ') and detected_region is not None:
            # Calibrate strings and frets
            if detector.calibrate_strings_and_frets(detected_region):
                print("✅ Guitar calibrated successfully!")
                guitar_detected = True
                break

        # Press 'q' to quit
        elif key == ord('q'):
            print("Calibration cancelled")
            break

    web_cam.release()
    cv2.destroyAllWindows()

    return detector if guitar_detected else None


def detect_string_and_fret(hand_landmark, frame_height, frame_width, guitar_detector):
    """
    Estimate which string and fret is being pressed based on hand position
    relative to the DETECTED guitar.
    Returns (string_number, fret_number, note)
    """
    # Get index finger tip position (landmark 8)
    if hand_landmark and len(hand_landmark.landmark) > 8:
        index_finger = hand_landmark.landmark[8]

        # Convert normalized coordinates to pixel coordinates
        pixel_x = int(index_finger.x * frame_width)
        pixel_y = int(index_finger.y * frame_height)

        # Use guitar detector to find string and fret
        string_num, fret_num = guitar_detector.get_string_and_fret_from_position(pixel_x, pixel_y)

        if string_num is not None and fret_num is not None:
            # Get the base note for this string
            base_note = GUITAR_STRINGS[string_num]

            # Get the note at this fret
            if fret_num < len(FRET_NOTES[base_note]):
                note = FRET_NOTES[base_note][fret_num]
                return string_num, fret_num, note

    return None, None, None


def record_audio_continuous(filename="test.wav", fs=44100):
    """Continuously record audio until stop_event is set."""
    print("🎙️  Audio recording started...")
    audio_buffer = []

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        audio_buffer.append(indata.copy())
        if stop_event.is_set():
            raise sd.CallbackStop()

    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        while not stop_event.is_set():
            sd.sleep(100)

    if audio_buffer:
        audio_data = np.concatenate(audio_buffer, axis=0)
        write(filename, fs, audio_data)
        print(f"💾 Audio saved as {filename}")
    return filename


def record_camera_with_tracking(guitar_detector):
    """
    Display camera, detect hands, and predict notes based on hand position
    relative to the detected guitar.
    """
    print("\n" + "="*60)
    print("🎬 RECORDING SESSION STARTED")
    print("="*60)
    print("📸 Camera active - tracking hands on guitar")
    print("🎙️  Audio recording in progress")
    print("⌨️  Press 'q' to stop and see results")
    print("="*60 + "\n")

    web_cam = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  # Detect both hands
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    start_time = time.time()
    frame_count = 0

    while not stop_event.is_set() and web_cam.isOpened():
        success, frame = web_cam.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Draw guitar overlay (strings and frets)
        frame = guitar_detector.draw_guitar_overlay(frame)

        # Process hand detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_note = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Detect which string and fret is being pressed on the DETECTED guitar
                string_num, fret_num, note = detect_string_and_fret(
                    hand_landmarks, h, w, guitar_detector
                )

                if note:
                    current_note = note
                    current_time = time.time() - start_time

                    # Only record every 5 frames to avoid duplicates
                    if frame_count % 5 == 0:
                        hand_positions.append({
                            "time_s": round(current_time, 3),
                            "string": string_num,
                            "fret": fret_num,
                            "predicted_note": note
                        })

                    # Display predicted note on screen
                    cv2.putText(frame, f"Playing: {note} (String {string_num+1}, Fret {fret_num})",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if not current_note:
            cv2.putText(frame, "Position hands on guitar fretboard",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # Show recording indicator
        elapsed = time.time() - start_time
        cv2.putText(frame, f"Recording: {int(elapsed)}s",
                   (w - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.circle(frame, (w - 200, 25), 5, (0, 0, 255), -1)  # Red recording dot

        cv2.imshow("GuitarMotion Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()

        frame_count += 1

    web_cam.release()
    cv2.destroyAllWindows()
    print("📷 Camera stopped.")


def analyze_audio(filename):
    """Analyze and detect notes from saved audio."""
    print("🔎 Analyzing recorded audio...")
    y, sr = lb.load(filename)
    f0 = lb.yin(y, fmin=lb.note_to_hz('E2'), fmax=lb.note_to_hz('E6'))
    note_names = lb.hz_to_note(f0)

    audio_notes = []
    for i, note in enumerate(note_names):
        if note != 'nan':
            timestamp = i * (len(y)/len(f0)) / sr
            audio_notes.append({"time_s": round(timestamp, 3), "note": note})

    return audio_notes


def compare_notes(hand_notes, audio_notes, tolerance=0.5):
    """
    Compare notes detected from hand position vs audio.
    Returns comparison results showing if user played correct notes.
    """
    print("\n🎯 Comparing hand positions with actual audio...")

    comparisons = []
    for hand_data in hand_notes:
        hand_time = hand_data["time_s"]
        predicted_note = hand_data["predicted_note"]

        # Find closest audio note in time
        closest_audio = None
        min_time_diff = float('inf')

        for audio_data in audio_notes:
            time_diff = abs(audio_data["time_s"] - hand_time)
            if time_diff < min_time_diff and time_diff < tolerance:
                min_time_diff = time_diff
                closest_audio = audio_data["note"]

        match = predicted_note == closest_audio if closest_audio else False

        comparisons.append({
            "time_s": hand_time,
            "predicted_note": predicted_note,
            "actual_note": closest_audio if closest_audio else "No audio detected",
            "match": match,
            "string": hand_data["string"],
            "fret": hand_data["fret"]
        })

    return comparisons


def save_session_data(hand_notes, audio_notes, comparisons):
    """Save comprehensive session data to JSON file."""
    # Ensure data directory exists
    data_dir = Path("data/recordings/sessions")
    data_dir.mkdir(parents=True, exist_ok=True)

    session_data = {
        "timestamp": int(time.time()),
        "hand_tracking": hand_notes,
        "audio_detection": audio_notes,
        "comparison": comparisons,
        "accuracy": calculate_accuracy(comparisons)
    }

    json_name = data_dir / f"session_{session_data['timestamp']}.json"
    with open(json_name, "w") as f:
        json.dump(session_data, f, indent=4)

    print(f"✅ Session saved as {json_name}")
    return json_name


def calculate_accuracy(comparisons):
    """Calculate how accurately the user played the intended notes."""
    if not comparisons:
        return 0.0

    matches = sum(1 for c in comparisons if c["match"])
    accuracy = (matches / len(comparisons)) * 100
    return round(accuracy, 2)


def print_results(comparisons):
    """Print comparison results in a readable format."""
    print("\n" + "="*60)
    print("📊 SESSION RESULTS")
    print("="*60)

    if not comparisons:
        print("No hand positions were detected during the session.")
        return

    accuracy = calculate_accuracy(comparisons)
    print(f"\n🎯 Overall Accuracy: {accuracy}%")
    print(f"✅ Correct notes: {sum(1 for c in comparisons if c['match'])}/{len(comparisons)}\n")

    print("Detailed comparison:")
    print("-" * 60)
    for i, comp in enumerate(comparisons[:10], 1):  # Show first 10
        match_icon = "✅" if comp["match"] else "❌"
        print(f"{match_icon} Time {comp['time_s']}s: "
              f"Predicted {comp['predicted_note']} (String {comp['string']+1}, Fret {comp['fret']}) | "
              f"Actual: {comp['actual_note']}")

    if len(comparisons) > 10:
        print(f"... and {len(comparisons) - 10} more")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎸 GUITAR MOTION TRACKER")
    print("="*60)
    print("This tool helps you learn guitar by:")
    print("  • Detecting your guitar using computer vision")
    print("  • Tracking your hand positions on the fretboard")
    print("  • Recording and analyzing the notes you play")
    print("  • Comparing hand positions with actual sound")
    print("  • Showing you if you're playing the right notes!")
    print("="*60 + "\n")

    # PHASE 1: Guitar Calibration
    print("PHASE 1: Guitar Detection & Calibration")
    guitar_detector = calibrate_guitar()

    if guitar_detector is None:
        print("❌ Guitar calibration failed or cancelled. Exiting.")
        exit(0)

    # Small delay for user to prepare
    print("\nGet ready to play! Starting in 3 seconds...")
    time.sleep(3)

    # PHASE 2: Recording Session
    print("\nPHASE 2: Recording Session")

    # Reset hand positions for this session
    hand_positions.clear()

    # Start audio recording in background
    audio_thread = threading.Thread(target=record_audio_continuous)
    audio_thread.start()

    # Run camera tracking on main thread
    record_camera_with_tracking(guitar_detector)

    # Wait for audio thread to finish
    audio_thread.join()

    # PHASE 3: Analysis & Results
    print("\nPHASE 3: Analysis & Results")

    # Analyze audio
    audio_notes = analyze_audio("test.wav")

    # Compare hand positions with audio
    comparisons = compare_notes(hand_positions, audio_notes)

    # Save all data
    save_session_data(hand_positions, audio_notes, comparisons)

    # Print results
    print_results(comparisons)

    print("✨ Session complete! Check data/recordings/sessions/ for detailed JSON logs.")
