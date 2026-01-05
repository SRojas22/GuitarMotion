"""
Lock State Manager
Manages the "Searching → Locking → Locked" state transitions
"""

from enum import Enum


class LockState(Enum):
    """Lock states for fretboard detection"""
    SEARCHING = "searching"   # No detection or low confidence
    LOCKING = "locking"       # Detections present, waiting for stability
    LOCKED = "locked"         # Stable, high-confidence detections


class LockStateManager:
    def __init__(self, lock_threshold=0.75, stable_frames=15):
        """
        Initialize lock state manager

        Args:
            lock_threshold: Minimum confidence to consider locking
            stable_frames: Number of consecutive detections needed to lock
        """
        self.state = LockState.SEARCHING
        self.lock_threshold = lock_threshold
        self.stable_frames = stable_frames
        self.consecutive_detections = 0

    def update(self, bbox, confidence):
        """
        Update state based on current detection

        Args:
            bbox: Detected bounding box or None
            confidence: Detection confidence 0-1

        Returns:
            current_state: LockState enum value
        """
        if bbox is None or confidence < self.lock_threshold:
            # Lost detection - go back to searching
            self.consecutive_detections = 0
            self.state = LockState.SEARCHING
        else:
            # Good detection - increment counter
            self.consecutive_detections += 1

            if self.consecutive_detections < self.stable_frames:
                # Still stabilizing
                self.state = LockState.LOCKING
            else:
                # Stable enough to lock
                self.state = LockState.LOCKED

        return self.state

    def reset(self):
        """Reset to searching state"""
        self.state = LockState.SEARCHING
        self.consecutive_detections = 0

    def is_locked(self):
        """Check if currently locked"""
        return self.state == LockState.LOCKED

    def get_lock_progress(self):
        """
        Get progress toward lock (useful for UI)

        Returns:
            float: 0.0 to 1.0 representing progress to lock
        """
        if self.state == LockState.LOCKED:
            return 1.0
        elif self.state == LockState.LOCKING:
            return self.consecutive_detections / self.stable_frames
        else:
            return 0.0
