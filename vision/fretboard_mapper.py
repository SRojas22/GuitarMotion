"""
Fretboard Geometry Mapper
Maps pixel coordinates to string/fret positions using real guitar math
"""

import cv2
import numpy as np


class FretboardMapper:
    def __init__(self, bbox, num_strings=6, num_frets=20):
        """
        Initialize fretboard mapper with detected bounding box

        Args:
            bbox: [x1, y1, x2, y2] bounding box from detector
            num_strings: Number of guitar strings (default 6)
            num_frets: Number of frets to map (default 12)
        """
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.num_strings = num_strings
        self.num_frets = num_frets

        # Calibration points (set by user clicks)
        self.nut_x = None  # X position of nut (0th fret)
        self.fret_12_x = None  # X position of 12th fret

        # Calculated positions
        self.string_positions = []  # Y coordinates for each string
        self.fret_positions = []  # X coordinates for each fret

        self.is_calibrated = False

    def set_reference_points(self, nut_x, fret_12_x):
        """
        Set calibration reference points from user clicks

        Args:
            nut_x: X coordinate of nut (0th fret)
            fret_12_x: X coordinate of 12th fret
        """
        self.nut_x = nut_x
        self.fret_12_x = fret_12_x
        self._calculate_geometry()
        self.is_calibrated = True
        print(f"✅ Fretboard calibrated: nut={nut_x}, 12th fret={fret_12_x}")

    def _calculate_geometry(self):
        """
        Calculate all string and fret positions using real guitar math
        """
        x1, y1, x2, y2 = map(int, self.bbox)

        # Calculate string positions (evenly spaced across bbox height)
        self.string_positions = []
        for i in range(self.num_strings):
            # Space strings evenly with small margins
            string_y = y1 + int((i + 0.5) * (y2 - y1) / self.num_strings)
            self.string_positions.append(string_y)

        # Calculate fret positions using logarithmic spacing
        # Guitar fret spacing formula: scale_length - (scale_length / 2^(fret/12))
        # Distance from nut to 12th fret = half the scale length
        scale_length = (self.fret_12_x - self.nut_x) * 2

        self.fret_positions = []
        for fret_num in range(self.num_frets + 1):  # Include nut (0) to 12th fret
            if fret_num == 0:
                fret_x = self.nut_x
            else:
                # Real guitar fret spacing formula
                fret_x = self.nut_x + (scale_length - scale_length / (2 ** (fret_num / 12)))
            self.fret_positions.append(int(fret_x))

    def get_string_fret_from_point(self, x, y):
        """
        Map pixel coordinates to (string_num, fret_num)

        Args:
            x, y: Pixel coordinates

        Returns:
            (string_num, fret_num): Zero-indexed positions, or (None, None) if out of range
        """
        if not self.is_calibrated:
            return None, None

        # Find nearest string (Y-axis)
        string_num = None
        min_dist = float('inf')
        for i, sy in enumerate(self.string_positions):
            dist = abs(y - sy)
            if dist < min_dist:
                min_dist = dist
                string_num = i

        # Reject if too far from any string
        string_threshold = (self.bbox[3] - self.bbox[1]) / self.num_strings / 2
        if min_dist > string_threshold:
            return None, None

        # Find which fret region the X coordinate falls into
        fret_num = None

        # Check if before nut
        if x < self.nut_x:
            return string_num, None

        # Check between frets
        for i in range(len(self.fret_positions) - 1):
            fret_start = self.fret_positions[i]
            fret_end = self.fret_positions[i + 1]

            if fret_start <= x < fret_end:
                fret_num = i
                break

        # Check if after last fret
        if fret_num is None and x >= self.fret_positions[-1]:
            fret_num = len(self.fret_positions) - 1

        return string_num, fret_num

    def draw_overlay(self, frame, string_color=(255, 200, 100), fret_color=(200, 200, 255), alpha=0.5):
        """
        Draw string and fret grid overlay on frame

        Args:
            frame: OpenCV frame to draw on
            string_color: RGB color for strings (default: cyan)
            fret_color: RGB color for frets (default: light blue)
            alpha: Transparency (0.0 to 1.0)

        Returns:
            frame: Frame with overlay drawn
        """
        if not self.is_calibrated:
            return frame

        overlay = frame.copy()
        x1, y1, x2, y2 = map(int, self.bbox)

        # Draw strings (horizontal lines)
        for i, sy in enumerate(self.string_positions):
            thickness = 2 if i in [0, 5] else 1  # Thicker for E strings
            cv2.line(overlay, (x1, sy), (x2, sy), string_color, thickness)

        # Draw frets (vertical lines) with labels
        for i, fx in enumerate(self.fret_positions):
            # Make nut (0), 3rd, 5th, 7th, 9th, 12th frets thicker
            thickness = 3 if i in [0, 3, 5, 7, 9, 12] else 1
            cv2.line(overlay, (fx, y1), (fx, y2), fret_color, thickness)

            # Draw fret number label (every other fret to avoid clutter)
            if i % 2 == 0 or i in [0, 3, 5, 7, 9, 12, 15, 17, 19]:
                label_y = y2 + 15  # Below the fretboard
                cv2.putText(overlay, str(i), (fx - 8, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, fret_color, 1)

        # Draw fret markers (dots) - standard guitar inlays
        # Single dots: 3, 5, 7, 9, 15, 17, 19
        # Double dots: 12
        marker_frets = {3: 1, 5: 1, 7: 1, 9: 1, 12: 2, 15: 1, 17: 1, 19: 1}
        for fret_num, num_dots in marker_frets.items():
            if fret_num < len(self.fret_positions):
                # Get position between this fret and next
                if fret_num < len(self.fret_positions) - 1:
                    fret_x = (self.fret_positions[fret_num] + self.fret_positions[fret_num + 1]) // 2
                else:
                    fret_x = self.fret_positions[fret_num]

                if num_dots == 1:
                    # Single dot in middle
                    center_y = (y1 + y2) // 2
                    cv2.circle(overlay, (fret_x, center_y), 6, (255, 255, 255), -1)
                    cv2.circle(overlay, (fret_x, center_y), 6, fret_color, 2)
                else:
                    # Double dots (12th fret)
                    upper_y = y1 + (y2 - y1) // 3
                    lower_y = y1 + 2 * (y2 - y1) // 3
                    cv2.circle(overlay, (fret_x, upper_y), 6, (255, 255, 255), -1)
                    cv2.circle(overlay, (fret_x, upper_y), 6, fret_color, 2)
                    cv2.circle(overlay, (fret_x, lower_y), 6, (255, 255, 255), -1)
                    cv2.circle(overlay, (fret_x, lower_y), 6, fret_color, 2)

        # Blend overlay with original frame
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    def draw_position_highlight(self, frame, string_num, fret_num, color=(0, 255, 0)):
        """
        Highlight a specific string/fret position

        Args:
            frame: OpenCV frame
            string_num: String index (0-5)
            fret_num: Fret index (0-12)
            color: RGB color for highlight

        Returns:
            frame: Frame with highlight drawn
        """
        if not self.is_calibrated:
            return frame

        if string_num is None or fret_num is None:
            return frame

        if string_num >= len(self.string_positions):
            return frame

        if fret_num >= len(self.fret_positions):
            return frame

        # Get position
        string_y = self.string_positions[string_num]

        # Get X position (center between fret and next fret)
        if fret_num < len(self.fret_positions) - 1:
            fret_x = (self.fret_positions[fret_num] + self.fret_positions[fret_num + 1]) // 2
        else:
            fret_x = self.fret_positions[fret_num]

        # Draw pulsing circle
        overlay = frame.copy()
        cv2.circle(overlay, (fret_x, string_y), 15, color, 3)
        cv2.circle(overlay, (fret_x, string_y), 10, color, -1)

        return cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    def get_bbox(self):
        """Get the bounding box"""
        return self.bbox

    def update_bbox(self, new_bbox):
        """
        Update bbox if fretboard moves (continuous tracking)
        Recalculates string positions but keeps fret spacing ratio

        Args:
            new_bbox: New [x1, y1, x2, y2] from detector
        """
        if not self.is_calibrated:
            return

        # Calculate scale factor
        old_width = self.bbox[2] - self.bbox[0]
        new_width = new_bbox[2] - new_bbox[0]
        scale_x = new_width / old_width

        # Update bbox
        old_x1 = self.bbox[0]
        self.bbox = new_bbox

        # Scale fret positions
        new_nut_x = new_bbox[0] + (self.nut_x - old_x1) * scale_x
        new_fret_12_x = new_bbox[0] + (self.fret_12_x - old_x1) * scale_x

        self.nut_x = int(new_nut_x)
        self.fret_12_x = int(new_fret_12_x)

        # Recalculate geometry
        self._calculate_geometry()
