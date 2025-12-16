"""
Guitar Detection Module
Uses computer vision to detect and track guitar fretboard in real-time
"""

import cv2
import numpy as np


class GuitarDetector:
    def __init__(self):
        self.guitar_region = None  # (x, y, width, height)
        self.fretboard_region = None
        self.string_positions = []  # Y-coordinates of 6 strings
        self.fret_positions = []    # X-coordinates of frets
        self.is_calibrated = False

    def detect_guitar_neck(self, frame):
        """
        Detect guitar neck/fretboard using edge detection and contour analysis.
        Returns the bounding box of the detected guitar neck.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Look for elongated rectangular shapes (guitar neck)
        best_candidate = None
        max_area = 0

        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Filter by aspect ratio (guitar neck is elongated)
            # Typical guitar neck in frame: width should be greater than height
            aspect_ratio = w / h if h > 0 else 0

            # Guitar neck should be reasonably wide and horizontal
            if aspect_ratio > 2.5 and area > 10000:  # Minimum area threshold
                if area > max_area:
                    max_area = area
                    best_candidate = (x, y, w, h)

        return best_candidate

    def detect_fretboard_color(self, frame):
        """
        Alternative method: Detect fretboard based on wood color (brown/tan tones).
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define range for brown/wood colors (typical fretboard colors)
        # This can be adjusted based on the actual guitar color
        lower_brown = np.array([5, 30, 30])
        upper_brown = np.array([25, 255, 255])

        # Create mask
        mask = cv2.inRange(hsv, lower_brown, upper_brown)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Check if it's elongated (guitar neck shape)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > 2.5:
                return (x, y, w, h)

        return None

    def detect_guitar(self, frame):
        """
        Combined detection: Try multiple methods to find the guitar.
        """
        # Try edge-based detection first
        guitar_region = self.detect_guitar_neck(frame)

        # If that fails, try color-based detection
        if guitar_region is None:
            guitar_region = self.detect_fretboard_color(frame)

        return guitar_region

    def calibrate_strings_and_frets(self, guitar_region, num_strings=6, num_frets=12):
        """
        Once guitar is detected, calculate positions of strings and frets.
        """
        if guitar_region is None:
            return False

        x, y, w, h = guitar_region

        # Calculate string positions (horizontal lines across the height)
        self.string_positions = []
        for i in range(num_strings):
            # Evenly space strings across the guitar neck height
            string_y = y + int((i + 0.5) * h / num_strings)
            self.string_positions.append(string_y)

        # Calculate fret positions (vertical lines across the width)
        self.fret_positions = []
        for i in range(num_frets + 1):  # +1 for open string
            # Frets get closer together as you go up the neck (logarithmic spacing)
            # For simplicity, using linear spacing here
            fret_x = x + int(i * w / num_frets)
            self.fret_positions.append(fret_x)

        self.guitar_region = guitar_region
        self.is_calibrated = True
        return True

    def get_string_and_fret_from_position(self, x, y):
        """
        Given an (x, y) coordinate, determine which string and fret it corresponds to.
        Returns (string_number, fret_number) or (None, None) if outside guitar region.
        """
        if not self.is_calibrated or self.guitar_region is None:
            return None, None

        gx, gy, gw, gh = self.guitar_region

        # Check if point is within guitar region
        if not (gx <= x <= gx + gw and gy <= y <= gy + gh):
            return None, None

        # Find closest string
        string_num = None
        min_string_dist = float('inf')
        for i, string_y in enumerate(self.string_positions):
            dist = abs(y - string_y)
            if dist < min_string_dist:
                min_string_dist = dist
                string_num = i

        # Find closest fret
        fret_num = None
        min_fret_dist = float('inf')
        for i, fret_x in enumerate(self.fret_positions):
            dist = abs(x - fret_x)
            if dist < min_fret_dist:
                min_fret_dist = dist
                fret_num = i

        # Only return if within reasonable distance to a string/fret
        string_threshold = gh / (2 * len(self.string_positions))  # Half the space between strings
        fret_threshold = gw / (2 * len(self.fret_positions))  # Half the space between frets

        if min_string_dist > string_threshold or min_fret_dist > fret_threshold:
            return None, None

        return string_num, fret_num

    def draw_guitar_overlay(self, frame):
        """
        Draw visual overlay showing detected guitar, strings, and frets.
        """
        if not self.is_calibrated or self.guitar_region is None:
            return frame

        overlay = frame.copy()
        x, y, w, h = self.guitar_region

        # Draw guitar region boundary (green box)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(overlay, "Guitar Detected", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw strings (horizontal lines)
        for i, string_y in enumerate(self.string_positions):
            cv2.line(overlay, (x, string_y), (x + w, string_y), (255, 200, 0), 1)
            cv2.putText(overlay, f"S{i+1}", (x - 25, string_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

        # Draw frets (vertical lines)
        for i, fret_x in enumerate(self.fret_positions):
            cv2.line(overlay, (fret_x, y), (fret_x, y + h), (200, 200, 255), 1)
            if i < len(self.fret_positions) - 1:
                cv2.putText(overlay, f"F{i}", (fret_x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 255), 1)

        # Blend overlay with original frame
        alpha = 0.7
        result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        return result

    def draw_detection_hint(self, frame, detected_region=None):
        """
        Draw helpful hints for positioning the guitar during calibration.
        """
        h, w = frame.shape[:2]

        if detected_region:
            # Guitar detected - show confirmation
            cv2.putText(frame, "Guitar Detected!", (w//2 - 100, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to confirm and start", (w//2 - 180, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw detected region
            x, y, w_box, h_box = detected_region
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        else:
            # No guitar detected - show instructions
            cv2.putText(frame, "Position guitar fretboard in frame", (w//2 - 200, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Guitar should be horizontal", (w//2 - 150, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "Keep fretboard visible and well-lit", (w//2 - 180, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Draw suggested region
            suggested_x = w // 6
            suggested_y = h // 3
            suggested_w = 2 * w // 3
            suggested_h = h // 3
            cv2.rectangle(frame, (suggested_x, suggested_y),
                         (suggested_x + suggested_w, suggested_y + suggested_h),
                         (255, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, "Press 'q' to quit", (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame
