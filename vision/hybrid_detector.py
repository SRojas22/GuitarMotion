"""
Hybrid Fretboard Detector
Combines YOLO detection with edge-based fallback for robustness
"""

from models.fretboard_detector import FretboardDetector
from src.guitar_detection import GuitarDetector


class HybridDetector:
    def __init__(self, model_path='models/weights/fretboard_detector.pt'):
        """
        Initialize hybrid detector with YOLO + edge-based fallback

        Args:
            model_path: Path to trained YOLOv8 model
        """
        self.yolo_detector = FretboardDetector(model_path=model_path)
        self.edge_detector = GuitarDetector()  # Legacy edge-based detector

    def detect(self, frame):
        """
        Try YOLO first, fallback to edge detection if needed

        Args:
            frame: OpenCV frame (BGR image)

        Returns:
            bbox: [x1, y1, x2, y2] or None
            confidence: float 0-1
            method: 'yolo' or 'edge' or None
        """
        # Try YOLO first (preferred method)
        bbox, conf = self.yolo_detector.detect_smoothed(frame)

        if bbox is not None:
            return bbox, conf, 'yolo'

        # YOLO failed - try edge-based fallback
        edge_bbox = self.edge_detector.detect_guitar(frame)

        if edge_bbox is not None:
            # Convert (x, y, w, h) to (x1, y1, x2, y2)
            x, y, w, h = edge_bbox
            bbox = [x, y, x + w, y + h]
            # Assign moderate confidence for edge detection
            return bbox, 0.6, 'edge'

        # Both methods failed
        return None, 0.0, None

    def reset(self):
        """Reset both detectors"""
        self.yolo_detector.reset_smoothing()
        # Edge detector doesn't need reset
