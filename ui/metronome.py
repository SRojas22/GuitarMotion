"""
Metronome and Song Timeline UI
Visual beat tracking and chord timeline display
"""

import cv2
import numpy as np


def draw_metronome_bar(frame, beat_in_bar, beats_per_bar=4, position='bottom'):
    """
    Draw visual beat indicator showing current beat

    Args:
        frame: OpenCV frame
        beat_in_bar: Current beat (1-indexed)
        beats_per_bar: Total beats per bar
        position: 'top' or 'bottom'

    Returns:
        frame: Frame with beat bar drawn
    """
    h, w = frame.shape[:2]

    bar_width = 400
    bar_height = 20
    x_start = (w - bar_width) // 2

    if position == 'bottom':
        y_pos = h - 60
    else:
        y_pos = 70

    # Background
    cv2.rectangle(frame, (x_start, y_pos),
                  (x_start + bar_width, y_pos + bar_height),
                  (30, 30, 30), -1)

    # Draw beat boxes
    beat_width = bar_width // beats_per_bar

    for i in range(beats_per_bar):
        x = x_start + i * beat_width

        # Color based on whether this beat is active
        if i < beat_in_bar:
            # First beat is accented
            if i == 0:
                color = (0, 255, 100)  # Bright green for downbeat
            else:
                color = (0, 200, 255)  # Yellow for other beats
        else:
            color = (80, 80, 80)  # Gray for inactive

        # Draw beat box
        cv2.rectangle(frame, (x + 2, y_pos + 2),
                     (x + beat_width - 2, y_pos + bar_height - 2),
                     color, -1)

        # Draw beat number
        cv2.putText(frame, str(i + 1), (x + beat_width // 2 - 5, y_pos + bar_height - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw border
    cv2.rectangle(frame, (x_start, y_pos),
                  (x_start + bar_width, y_pos + bar_height),
                  (200, 200, 200), 1)

    return frame


def draw_chord_timeline(frame, upcoming_chords, current_bar, position='bottom-left'):
    """
    Show upcoming chord changes

    Args:
        frame: OpenCV frame
        upcoming_chords: List of {chord, bar} dicts
        current_bar: Current bar number
        position: Where to draw timeline

    Returns:
        frame: Frame with chord timeline
    """
    h, w = frame.shape[:2]

    if position == 'bottom-left':
        x_start = 20
        y_start = h - 140
    elif position == 'top-left':
        x_start = 20
        y_start = 80
    else:
        x_start = 20
        y_start = h - 140

    # Draw title
    cv2.putText(frame, "Upcoming:", (x_start, y_start - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Draw each upcoming chord
    for i, event in enumerate(upcoming_chords[:3]):  # Show next 3 chords
        y = y_start + i * 30

        # Highlight current chord
        if event['bar'] == current_bar:
            color = (0, 255, 255)  # Bright yellow for current
            box_color = (0, 200, 200)
            font_scale = 0.7
        elif event['bar'] == current_bar + 1:
            color = (0, 255, 150)  # Green for next
            box_color = None
            font_scale = 0.6
        else:
            color = (150, 150, 150)  # Gray for later
            box_color = None
            font_scale = 0.5

        # Draw background box for current chord
        if box_color:
            overlay = frame.copy()
            cv2.rectangle(overlay, (x_start - 5, y - 20),
                         (x_start + 180, y + 5), box_color, -1)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        # Draw chord name and bar
        text = f"{event['chord']} - Bar {event['bar']}"
        cv2.putText(frame, text, (x_start, y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2 if box_color else 1)

    return frame


def draw_song_progress(frame, current_bar, total_bars, progress_pct, position='top'):
    """
    Draw song progress bar

    Args:
        frame: OpenCV frame
        current_bar: Current bar number
        total_bars: Total bars in song
        progress_pct: Progress as 0.0-1.0

    Returns:
        frame: Frame with progress bar
    """
    h, w = frame.shape[:2]

    bar_width = 500
    bar_height = 12
    x_start = (w - bar_width) // 2

    if position == 'top':
        y_pos = 20
    else:
        y_pos = h - 20

    # Background
    cv2.rectangle(frame, (x_start, y_pos),
                  (x_start + bar_width, y_pos + bar_height),
                  (40, 40, 40), -1)

    # Progress
    progress_width = int(bar_width * progress_pct)
    if progress_width > 0:
        cv2.rectangle(frame, (x_start, y_pos),
                     (x_start + progress_width, y_pos + bar_height),
                     (0, 200, 255), -1)

    # Border
    cv2.rectangle(frame, (x_start, y_pos),
                  (x_start + bar_width, y_pos + bar_height),
                  (200, 200, 200), 1)

    # Text (bar count)
    text = f"Bar {current_bar}/{total_bars}"
    cv2.putText(frame, text, (x_start + bar_width + 10, y_pos + 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


def draw_song_info(frame, song_title, artist, bpm, position='top-left'):
    """
    Draw song information banner

    Args:
        frame: OpenCV frame
        song_title: Song title
        artist: Artist name
        bpm: Beats per minute

    Returns:
        frame: Frame with song info
    """
    h, w = frame.shape[:2]

    if position == 'top-left':
        x, y = 10, 10
    else:
        x, y = 10, 10

    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + 400, y + 60), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    # Draw border
    cv2.rectangle(frame, (x, y), (x + 400, y + 60), (0, 200, 255), 2)

    # Draw song title
    cv2.putText(frame, song_title, (x + 10, y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw artist and BPM
    info_text = f"{artist} • {bpm} BPM"
    cv2.putText(frame, info_text, (x + 10, y + 48),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


def draw_streak_counter(frame, streak, max_streak, position='top-right'):
    """
    Draw streak counter (consecutive correct chords)

    Args:
        frame: OpenCV frame
        streak: Current streak
        max_streak: Best streak this session

    Returns:
        frame: Frame with streak counter
    """
    h, w = frame.shape[:2]

    if position == 'top-right':
        x = w - 160
        y = 80
    else:
        x = w - 160
        y = 80

    # Determine color based on streak
    if streak >= 10:
        color = (0, 255, 0)  # Green
        emoji = "🔥"
    elif streak >= 5:
        color = (0, 255, 255)  # Yellow
        emoji = "⭐"
    else:
        color = (200, 200, 200)  # Gray
        emoji = ""

    # Draw background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 10, y - 30), (x + 150, y + 20), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.rectangle(frame, (x - 10, y - 30), (x + 150, y + 20), color, 2)

    # Draw streak
    streak_text = f"Streak: {streak}"
    cv2.putText(frame, streak_text, (x, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Draw max streak
    max_text = f"Best: {max_streak}"
    cv2.putText(frame, max_text, (x, y + 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    return frame
