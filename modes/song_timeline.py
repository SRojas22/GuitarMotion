"""
Song Timeline Manager
Manages progression through a song's chord changes
"""

import json
import time
from pathlib import Path


class SongTimeline:
    def __init__(self, song_data):
        """
        Initialize song timeline from song data

        Args:
            song_data: Dictionary with song info and timeline
        """
        self.song = song_data
        self.bpm = song_data['bpm']
        self.time_signature = song_data.get('time_signature', '4/4')
        self.beats_per_bar = int(self.time_signature.split('/')[0])

        # Calculate timing
        self.beat_duration = 60.0 / self.bpm  # seconds per beat
        self.bar_duration = self.beat_duration * self.beats_per_bar

        self.timeline = sorted(song_data['timeline'], key=lambda x: x['bar'])
        self.start_time = None
        self.paused = False
        self.pause_time = 0

    def start(self):
        """Start the timeline"""
        self.start_time = time.time()
        self.paused = False
        print(f"🎵 Starting '{self.song['title']}' at {self.bpm} BPM")

    def pause(self):
        """Pause the timeline"""
        if not self.paused:
            self.pause_time = time.time()
            self.paused = True

    def resume(self):
        """Resume from pause"""
        if self.paused:
            pause_duration = time.time() - self.pause_time
            self.start_time += pause_duration
            self.paused = False

    def get_elapsed_time(self):
        """Get elapsed time in seconds"""
        if not self.start_time:
            return 0
        if self.paused:
            return self.pause_time - self.start_time
        return time.time() - self.start_time

    def get_current_bar(self):
        """Get current bar number (1-indexed)"""
        elapsed = self.get_elapsed_time()
        return int(elapsed / self.bar_duration) + 1

    def get_current_beat(self):
        """Get current beat within bar (1-indexed)"""
        elapsed = self.get_elapsed_time()
        beat_in_bar = int((elapsed % self.bar_duration) / self.beat_duration) + 1
        return min(beat_in_bar, self.beats_per_bar)

    def get_current_chord(self):
        """
        Get chord that should be played at current time

        Returns:
            str: Chord name or None
        """
        if not self.start_time:
            return None

        current_bar = self.get_current_bar()

        # Find the most recent chord change at or before current bar
        current_chord = None
        for event in self.timeline:
            if event['bar'] <= current_bar:
                current_chord = event['chord']
            else:
                break

        return current_chord

    def get_next_chord(self):
        """
        Get the next upcoming chord

        Returns:
            tuple: (chord_name, bar_number) or (None, None)
        """
        if not self.start_time:
            return None, None

        current_bar = self.get_current_bar()

        # Find next chord change after current bar
        for event in self.timeline:
            if event['bar'] > current_bar:
                return event['chord'], event['bar']

        return None, None

    def get_upcoming_chords(self, n=3):
        """
        Get next N upcoming chord changes

        Args:
            n: Number of upcoming chords to get

        Returns:
            list: List of {chord, bar} dicts
        """
        if not self.start_time:
            return []

        current_bar = self.get_current_bar()

        upcoming = []
        for event in self.timeline:
            if event['bar'] >= current_bar:
                upcoming.append(event)
            if len(upcoming) >= n:
                break

        return upcoming

    def get_progress(self):
        """
        Get song progress

        Returns:
            tuple: (current_bar, current_beat, total_bars, progress_percentage)
        """
        current_bar = self.get_current_bar()
        current_beat = self.get_current_beat()
        total_bars = max(e['bar'] for e in self.timeline) if self.timeline else 0

        if total_bars > 0:
            progress = min(current_bar / total_bars, 1.0)
        else:
            progress = 0.0

        return current_bar, current_beat, total_bars, progress

    def is_finished(self):
        """Check if song has finished"""
        if not self.start_time:
            return False

        current_bar = self.get_current_bar()
        total_bars = max(e['bar'] for e in self.timeline) if self.timeline else 0

        return current_bar > total_bars

    @staticmethod
    def load_from_file(song_path):
        """
        Load song from JSON file

        Args:
            song_path: Path to song JSON file

        Returns:
            SongTimeline: Initialized timeline
        """
        song_file = Path(song_path)
        if not song_file.exists():
            raise FileNotFoundError(f"Song file not found: {song_path}")

        with open(song_file, 'r') as f:
            song_data = json.load(f)

        return SongTimeline(song_data)

    @staticmethod
    def list_available_songs(songs_dir='data/songs'):
        """
        List all available songs

        Args:
            songs_dir: Directory containing song JSON files

        Returns:
            list: List of (filename, song_data) tuples
        """
        songs_path = Path(songs_dir)
        if not songs_path.exists():
            return []

        available = []
        for song_file in songs_path.glob('*.json'):
            try:
                with open(song_file, 'r') as f:
                    song_data = json.load(f)
                available.append((song_file, song_data))
            except Exception as e:
                print(f"⚠️  Couldn't load {song_file.name}: {e}")

        return available
