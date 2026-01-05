"""
Background Music Player
Plays song audio in the background with volume control
"""

import pygame
from pathlib import Path


class BackgroundMusicPlayer:
    def __init__(self):
        """Initialize pygame mixer for audio playback"""
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.volume = 0.3  # Default 30% volume
        self.is_playing = False
        self.current_file = None

    def load_song(self, audio_file):
        """
        Load an audio file

        Args:
            audio_file: Path to audio file (mp3, wav, ogg)

        Returns:
            bool: True if loaded successfully
        """
        audio_path = Path(audio_file)

        if not audio_path.exists():
            print(f"⚠️  Audio file not found: {audio_file}")
            return False

        try:
            pygame.mixer.music.load(str(audio_path))
            self.current_file = audio_file
            pygame.mixer.music.set_volume(self.volume)
            print(f"✅ Loaded audio: {audio_path.name}")
            return True
        except Exception as e:
            print(f"❌ Failed to load audio: {e}")
            return False

    def play(self, loops=0):
        """
        Start playing the loaded song

        Args:
            loops: Number of times to loop (-1 for infinite, 0 for once)
        """
        if self.current_file:
            try:
                pygame.mixer.music.play(loops=loops)
                self.is_playing = True
                print(f"▶️  Playing at {int(self.volume * 100)}% volume")
            except Exception as e:
                print(f"❌ Playback error: {e}")

    def stop(self):
        """Stop playback"""
        pygame.mixer.music.stop()
        self.is_playing = False
        print("⏹️  Stopped")

    def pause(self):
        """Pause playback"""
        pygame.mixer.music.pause()
        print("⏸️  Paused")

    def unpause(self):
        """Resume playback"""
        pygame.mixer.music.unpause()
        print("▶️  Resumed")

    def set_volume(self, volume):
        """
        Set playback volume

        Args:
            volume: Volume level 0.0 (mute) to 1.0 (max)
        """
        self.volume = max(0.0, min(1.0, volume))  # Clamp to 0-1
        pygame.mixer.music.set_volume(self.volume)
        print(f"🔊 Volume: {int(self.volume * 100)}%")

    def increase_volume(self, step=0.1):
        """Increase volume by step"""
        self.set_volume(self.volume + step)

    def decrease_volume(self, step=0.1):
        """Decrease volume by step"""
        self.set_volume(self.volume - step)

    def is_song_playing(self):
        """Check if music is currently playing"""
        return pygame.mixer.music.get_busy()

    def cleanup(self):
        """Clean up pygame mixer"""
        pygame.mixer.music.stop()
        pygame.mixer.quit()
