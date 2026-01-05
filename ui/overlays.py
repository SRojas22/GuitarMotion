"""
UI Overlay System
Beautiful visual overlays for detection states and feedback
"""

import cv2
import numpy as np
from vision.lock_state import LockState


def draw_lock_state_ui(frame, state, confidence=0.0, method=None):
    """
    Draw lock state indicator in top-left corner

    Args:
        frame: OpenCV frame to draw on
        state: LockState enum value
        confidence: Detection confidence 0-1
        method: Detection method ('yolo', 'edge', or None)

    Returns:
        frame: Frame with lock state UI drawn
    """
    h, w = frame.shape[:2]

    # Determine color and text based on state
    if state == LockState.SEARCHING:
        color = (0, 165, 255)  # Orange
        text = "Searching for fretboard..."
    elif state == LockState.LOCKING:
        color = (0, 255, 255)  # Yellow
        text = f"Locking... {confidence:.0%}"
    else:  # LOCKED
        color = (0, 255, 0)  # Green
        text = "Locked ✅"

    # Add method indicator if available
    if method == 'yolo':
        text += " (AI)"
    elif method == 'edge':
        text += " (CV)"

    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 50), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    # Draw border with state color
    cv2.rectangle(frame, (10, 10), (300, 50), color, 2)

    # Draw text
    cv2.putText(frame, text, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


def draw_fretboard_glow(frame, bbox, color=(0, 255, 200)):
    """
    Draw neon glow effect around detected fretboard

    Args:
        frame: OpenCV frame to draw on
        bbox: [x1, y1, x2, y2] bounding box
        color: RGB color for glow (default: cyan)

    Returns:
        frame: Frame with glow effect
    """
    overlay = frame.copy()
    x1, y1, x2, y2 = map(int, bbox)

    # Outer glow (thicker, semi-transparent)
    for thickness in [10, 8, 6]:
        alpha = 0.2 + (10 - thickness) * 0.05
        cv2.rectangle(overlay, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                      color, thickness)

    # Inner sharp line
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

    # Blend
    return cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)


def draw_progress_bar(frame, progress, position='bottom'):
    """
    Draw progress bar (for locking progress)

    Args:
        frame: OpenCV frame to draw on
        progress: float 0.0 to 1.0
        position: 'top' or 'bottom'

    Returns:
        frame: Frame with progress bar
    """
    h, w = frame.shape[:2]

    bar_width = 400
    bar_height = 8
    x_start = (w - bar_width) // 2

    if position == 'bottom':
        y_pos = h - 30
    else:
        y_pos = 70

    # Background
    cv2.rectangle(frame, (x_start, y_pos),
                  (x_start + bar_width, y_pos + bar_height),
                  (50, 50, 50), -1)

    # Progress
    progress_width = int(bar_width * progress)
    if progress_width > 0:
        # Color changes from yellow to green as progress increases
        if progress < 0.5:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 255, 0)  # Green

        cv2.rectangle(frame, (x_start, y_pos),
                      (x_start + progress_width, y_pos + bar_height),
                      color, -1)

    return frame


def draw_instructions(frame, locked=False):
    """
    Draw helpful instructions for the user

    Args:
        frame: OpenCV frame to draw on
        locked: Whether fretboard is locked

    Returns:
        frame: Frame with instructions
    """
    h, w = frame.shape[:2]

    if not locked:
        instructions = [
            "Position your guitar fretboard in frame",
            "Press SPACE when locked to continue",
            "Press Q to quit"
        ]
        color = (200, 200, 200)
        start_y = h - 100
    else:
        instructions = [
            "Guitar locked! Press SPACE to start session",
            "Press Q to quit"
        ]
        color = (0, 255, 0)
        start_y = h - 80

    for i, instruction in enumerate(instructions):
        cv2.putText(frame, instruction,
                    (20, start_y + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame


def draw_detection_stats(frame, bbox, confidence, method):
    """
    Draw detection statistics (for debugging/info)

    Args:
        frame: OpenCV frame
        bbox: Bounding box [x1, y1, x2, y2]
        confidence: Detection confidence
        method: Detection method

    Returns:
        frame: Frame with stats
    """
    if bbox is None:
        return frame

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    # Calculate bbox dimensions
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    stats = [
        f"Confidence: {confidence:.1%}",
        f"Method: {method.upper() if method else 'N/A'}",
        f"Size: {bbox_w}x{bbox_h}px"
    ]

    # Draw stats near bbox
    stats_x = min(x2 + 10, w - 200)
    stats_y = max(y1, 60)

    for i, stat in enumerate(stats):
        cv2.putText(frame, stat,
                    (stats_x, stats_y + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)

    return frame
