"""
GuitarMotion Modes Module
Different practice and learning modes for guitar
"""

from .chord_coach import ChordCoach
from .chord_coach_session import run_chord_coach, quick_chord_test
from .play_along_session import run_play_along
from .song_timeline import SongTimeline

__all__ = [
    'ChordCoach',
    'run_chord_coach',
    'quick_chord_test',
    'run_play_along',
    'SongTimeline'
]
