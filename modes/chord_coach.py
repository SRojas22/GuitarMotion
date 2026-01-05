"""
Chord Coach Module
Helps users learn and practice guitar chords with real-time feedback
"""

import json
import cv2
import numpy as np
from pathlib import Path


class ChordCoach:
    def __init__(self, chord_library_path='data/chords.json'):
        """
        Initialize chord coach with chord library

        Args:
            chord_library_path: Path to JSON file with chord definitions
        """
        self.chord_library_path = chord_library_path
        self.chords = self._load_chords()
        self.current_chord = None
        self.current_chord_name = None
        self.mapper = None

    def _load_chords(self):
        """Load chord library from JSON file"""
        chord_file = Path(self.chord_library_path)
        if not chord_file.exists():
            print(f"⚠️  Chord library not found at {chord_file}")
            return {}

        with open(chord_file, 'r') as f:
            chords = json.load(f)

        print(f"✅ Loaded {len(chords)} chords from library")
        return chords

    def get_available_chords(self):
        """Get list of available chord names"""
        return list(self.chords.keys())

    def select_chord(self, chord_name):
        """
        Set target chord to practice

        Args:
            chord_name: Name of chord (e.g., "C", "G", "Em")

        Returns:
            bool: True if chord found, False otherwise
        """
        if chord_name in self.chords:
            self.current_chord = self.chords[chord_name]
            self.current_chord_name = chord_name
            print(f"🎯 Selected chord: {self.current_chord['name']}")
            return True
        else:
            print(f"❌ Chord '{chord_name}' not found in library")
            return False

    def set_mapper(self, mapper):
        """
        Set fretboard mapper for position calculations

        Args:
            mapper: FretboardMapper instance (must be calibrated)
        """
        self.mapper = mapper

    def check_finger_placement(self, hand_landmarks, frame_shape):
        """
        Detect which string/fret positions user is pressing

        Args:
            hand_landmarks: MediaPipe hand landmarks
            frame_shape: (height, width, channels) of frame

        Returns:
            list: Detected finger positions [{string, fret, landmark_id}]
        """
        if not self.mapper or not hand_landmarks:
            return []

        h, w = frame_shape[:2]
        detected_fingers = []

        # Check fingertips: index(8), middle(12), ring(16), pinky(20)
        fingertip_landmarks = [8, 12, 16, 20]

        for tip_id in fingertip_landmarks:
            if len(hand_landmarks.landmark) > tip_id:
                finger_tip = hand_landmarks.landmark[tip_id]
                px = int(finger_tip.x * w)
                py = int(finger_tip.y * h)

                string_num, fret_num = self.mapper.get_string_fret_from_point(px, py)

                # Only count if finger is pressing a fret (not open strings)
                if string_num is not None and fret_num is not None and fret_num > 0:
                    detected_fingers.append({
                        'string': string_num,
                        'fret': fret_num,
                        'landmark_id': tip_id,
                        'position': (px, py)
                    })

        return detected_fingers

    def score_placement(self, detected_fingers):
        """
        Compare detected finger positions with target chord

        Args:
            detected_fingers: List of detected finger positions

        Returns:
            dict: Score with correct, missing, extra fingers and accuracy
        """
        if not self.current_chord:
            return None

        # Expected positions (string, fret)
        expected = set((f['string'], f['fret']) for f in self.current_chord['fingers'])

        # Detected positions (string, fret)
        detected = set((f['string'], f['fret']) for f in detected_fingers)

        # Calculate matches
        correct = expected & detected
        missing = expected - detected
        extra = detected - expected

        # Calculate accuracy
        if len(expected) > 0:
            accuracy = len(correct) / len(expected)
        else:
            accuracy = 1.0 if len(detected) == 0 else 0.0

        return {
            'correct': list(correct),
            'missing': list(missing),
            'extra': list(extra),
            'accuracy': accuracy,
            'num_correct': len(correct),
            'num_expected': len(expected)
        }

    def draw_target_overlay(self, frame, color=(100, 150, 255), alpha=0.7):
        """
        Draw ghost circles showing where to place fingers

        Args:
            frame: OpenCV frame
            color: RGB color for target circles
            alpha: Transparency (0.0 to 1.0)

        Returns:
            frame: Frame with target overlay
        """
        if not self.current_chord or not self.mapper:
            return frame

        overlay = frame.copy()

        for finger in self.current_chord['fingers']:
            string_num = finger['string']
            fret_num = finger['fret']

            # Get pixel coordinates from mapper
            if string_num >= len(self.mapper.string_positions):
                continue
            if fret_num >= len(self.mapper.fret_positions):
                continue

            string_y = self.mapper.string_positions[string_num]

            # Get fret position (center between fret lines)
            if fret_num < len(self.mapper.fret_positions) - 1:
                fret_x = (self.mapper.fret_positions[fret_num] +
                         self.mapper.fret_positions[fret_num + 1]) // 2
            else:
                fret_x = self.mapper.fret_positions[fret_num]

            # Draw pulsing ghost circle
            cv2.circle(overlay, (fret_x, string_y), 18, color, 3)
            cv2.circle(overlay, (fret_x, string_y), 12, color, -1)

            # Draw fret number
            cv2.putText(overlay, str(fret_num), (fret_x - 8, string_y + 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Blend overlay
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    def draw_detected_fingers(self, frame, detected_fingers, score):
        """
        Draw circles on detected finger positions with color coding

        Args:
            frame: OpenCV frame
            detected_fingers: List of detected positions
            score: Score dict from score_placement()

        Returns:
            frame: Frame with detected fingers drawn
        """
        if not score:
            return frame

        overlay = frame.copy()

        for finger in detected_fingers:
            pos_tuple = (finger['string'], finger['fret'])
            px, py = finger['position']

            # Color based on correctness
            if pos_tuple in score['correct']:
                color = (0, 255, 0)  # Green - correct
            elif pos_tuple in score['extra']:
                color = (0, 0, 255)  # Red - wrong position
            else:
                color = (200, 200, 200)  # Gray - neutral

            # Draw circle
            cv2.circle(overlay, (px, py), 10, color, 2)
            cv2.circle(overlay, (px, py), 4, color, -1)

        return cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

    def draw_feedback(self, frame, score):
        """
        Show accuracy feedback and instructions

        Args:
            frame: OpenCV frame
            score: Score dict from score_placement()

        Returns:
            frame: Frame with feedback overlay
        """
        if not score:
            return frame

        h, w = frame.shape[:2]
        accuracy = score['accuracy']

        # Determine feedback message and color
        if accuracy >= 0.95:
            color = (0, 255, 0)  # Green
            text = "Perfect! ✓"
            emoji = ""
        elif accuracy >= 0.7:
            color = (0, 255, 255)  # Yellow
            text = f"Almost! {accuracy:.0%}"
            emoji = ""
        elif accuracy >= 0.3:
            color = (0, 165, 255)  # Orange
            text = f"Keep trying {accuracy:.0%}"
            emoji = ""
        else:
            color = (0, 100, 255)  # Red-orange
            text = "Check finger positions"
            emoji = ""

        # Draw feedback box
        overlay = frame.copy()
        box_height = 80
        cv2.rectangle(overlay, (10, 60), (w - 10, 60 + box_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.rectangle(frame, (10, 60), (w - 10, 60 + box_height), color, 2)

        # Main feedback text
        cv2.putText(frame, text, (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Finger count
        finger_text = f"{score['num_correct']}/{score['num_expected']} fingers"
        cv2.putText(frame, finger_text, (20, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Missing fingers hint
        if score['missing']:
            missing_text = f"Missing: {len(score['missing'])} finger(s)"
            cv2.putText(frame, missing_text, (w - 250, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        # Extra fingers hint
        if score['extra']:
            extra_text = f"Extra: {len(score['extra'])} finger(s)"
            cv2.putText(frame, extra_text, (w - 250, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

        return frame

    def draw_chord_diagram(self, frame, position='top-right'):
        """
        Draw a small chord diagram in the corner

        Args:
            frame: OpenCV frame
            position: 'top-right', 'top-left', etc.

        Returns:
            frame: Frame with chord diagram
        """
        if not self.current_chord:
            return frame

        h, w = frame.shape[:2]

        # Chord name display
        if position == 'top-right':
            x, y = w - 200, 10
        else:
            x, y = 10, 10

        # Draw chord name banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + 190, y + 40), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        cv2.rectangle(frame, (x, y), (x + 190, y + 40), (255, 255, 255), 2)
        cv2.putText(frame, self.current_chord['name'], (x + 10, y + 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame
