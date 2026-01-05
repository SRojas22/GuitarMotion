"""
YOLO-based Fretboard Detector
Uses trained YOLOv8 model to detect guitar fretboards in real-time
"""

from ultralytics import YOLO
import numpy as np


class FretboardDetector:
    def __init__(self, model_path='models/weights/fretboard_detector.pt',
                 confidence_threshold=0.7):
        """
        Initialize fretboard detector with trained YOLO model

        Args:
            model_path: Path to trained YOLOv8 model weights
            confidence_threshold: Minimum confidence for valid detection
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.bbox_history = []  # For EMA smoothing
        self.alpha = 0.3  # EMA weight (lower = smoother but slower response)

    def detect_raw(self, frame):
        """
        Run YOLO inference on frame

        Args:
            frame: OpenCV frame (BGR image)

        Returns:
            bbox: [x1, y1, x2, y2] or None
            confidence: float 0-1 or 0.0
        """
        # Run inference
        results = self.model(frame, verbose=False)

        if len(results[0].boxes) > 0:
            # Get highest confidence detection
            best_box = max(results[0].boxes, key=lambda x: x.conf)
            bbox = best_box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            conf = float(best_box.conf)
            return bbox, conf

        return None, 0.0

    def detect_smoothed(self, frame):
        """
        Apply EMA smoothing to reduce jitter

        Args:
            frame: OpenCV frame (BGR image)

        Returns:
            bbox: Smoothed [x1, y1, x2, y2] or None
            confidence: float 0-1 or 0.0
        """
        raw_bbox, conf = self.detect_raw(frame)

        if raw_bbox is not None and conf > self.confidence_threshold:
            if len(self.bbox_history) == 0:
                # First detection - use as is
                smoothed = raw_bbox
            else:
                # Apply exponential moving average
                prev = self.bbox_history[-1]
                smoothed = self.alpha * raw_bbox + (1 - self.alpha) * prev

            # Update history (keep last 10)
            self.bbox_history.append(smoothed)
            if len(self.bbox_history) > 10:
                self.bbox_history.pop(0)

            return smoothed, conf

        # No valid detection - clear history
        if conf < self.confidence_threshold:
            self.bbox_history.clear()

        return None, conf

    def reset_smoothing(self):
        """Clear smoothing history (useful when restarting detection)"""
        self.bbox_history.clear()
