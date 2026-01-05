"""
Play-Along Session
Play along with songs, get real-time feedback and scoring
"""

import cv2
import mediapipe as mp
import time
from pathlib import Path
from .chord_coach import ChordCoach
from .song_timeline import SongTimeline
from audio.background_music import BackgroundMusicPlayer
from ui.metronome import (
    draw_metronome_bar,
    draw_chord_timeline,
    draw_song_progress,
    draw_song_info,
    draw_streak_counter
)


def run_play_along(mapper, detector):
    """
    Main play-along session with song timeline

    Args:
        mapper: Calibrated FretboardMapper
        detector: HybridDetector instance

    Returns:
        bool: True if session completed successfully
    """
    print("\n" + "="*60)
    print("🎵 PLAY-ALONG MODE")
    print("="*60)
    print("Play along with your favorite songs!")
    print("="*60 + "\n")

    # List available songs
    available_songs = SongTimeline.list_available_songs()

    if not available_songs:
        print("❌ No songs found in data/songs/")
        return False

    print("Available songs:")
    for i, (song_file, song_data) in enumerate(available_songs, 1):
        print(f"  {i}. {song_data['title']} - {song_data['artist']} "
              f"({song_data['difficulty']}, {song_data['bpm']} BPM)")

    # Let user select song
    while True:
        try:
            choice = input("\nSelect song (1-{}): ".format(len(available_songs))).strip()
            song_idx = int(choice) - 1
            if 0 <= song_idx < len(available_songs):
                song_file, song_data = available_songs[song_idx]
                break
            else:
                print("⚠️  Invalid choice")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Cancelled")
            return False

    # Initialize song timeline
    timeline = SongTimeline(song_data)

    # Initialize chord coach for scoring
    coach = ChordCoach()
    coach.set_mapper(mapper)

    # Background music player - initialize early
    music_player = BackgroundMusicPlayer()
    music_enabled = False

    # Check if song has audio file
    audio_file = song_data.get('audio_file')
    if audio_file and Path(audio_file).exists():
        if music_player.load_song(audio_file):
            music_enabled = True
        else:
            print("⚠️  Couldn't load audio file")

    print(f"\n✅ Loaded: {song_data['title']}")
    print(f"   BPM: {song_data['bpm']}")
    print(f"   Total bars: {max(e['bar'] for e in timeline.timeline)}")
    print("\n" + "="*60)
    print("Instructions:")
    print("  • Song will start in 3 seconds")
    print("  • Follow the chord changes")
    print("  • Green = correct chord, Red = wrong chord")
    print("  • Build streaks for consecutive correct chords!")
    print("\nControls:")
    print("  • Q - Quit")
    if music_enabled:
        print("  • M - Toggle music pause/play")
        print("  • + - Increase volume")
        print("  • - - Decrease volume")
    print("="*60 + "\n")

    # Setup hand tracking
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # Setup camera
    web_cam = cv2.VideoCapture(0)

    if not web_cam.isOpened():
        print("❌ Failed to open camera")
        return False

    # Session stats
    streak = 0
    max_streak = 0
    scores = []  # Accuracy per frame
    chord_scores = {}  # Per-chord accuracy

    # Countdown
    print("Get ready...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    print("🎵 GO!\n")
    timeline.start()

    # Start background music if enabled
    if music_enabled:
        music_player.play()

    while web_cam.isOpened():
        success, frame = web_cam.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Get current song position
        current_bar, current_beat, total_bars, progress = timeline.get_progress()

        # Check if song finished
        if timeline.is_finished():
            print("\n🎉 Song complete!")
            break

        # Get current and upcoming chords
        current_chord = timeline.get_current_chord()
        upcoming = timeline.get_upcoming_chords(n=3)

        # Draw fretboard overlay
        frame = mapper.draw_overlay(frame, alpha=0.3)

        # Select current chord in coach
        if current_chord:
            if coach.current_chord_name != current_chord:
                coach.select_chord(current_chord)

            # Draw target chord overlay
            frame = coach.draw_target_overlay(frame, alpha=0.5)

        # Detect hands and score
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        score = None
        if results.multi_hand_landmarks and current_chord:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 200), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2)
                )

                # Check finger placement
                detected_fingers = coach.check_finger_placement(
                    hand_landmarks, (h, w, 3)
                )

                # Score placement
                score = coach.score_placement(detected_fingers)

                # Draw detected fingers
                if score:
                    frame = coach.draw_detected_fingers(frame, detected_fingers, score)

        # Update streak
        if score:
            accuracy = score['accuracy']
            scores.append(accuracy)

            # Track per-chord accuracy
            if current_chord not in chord_scores:
                chord_scores[current_chord] = []
            chord_scores[current_chord].append(accuracy)

            if accuracy >= 0.8:  # 80% threshold for streak
                streak += 1
                max_streak = max(streak, max_streak)
            else:
                streak = 0
        else:
            scores.append(0)
            streak = 0

        # Draw song UI
        frame = draw_song_info(frame, song_data['title'], song_data['artist'],
                              song_data['bpm'], position='top-left')

        frame = draw_song_progress(frame, current_bar, total_bars, progress,
                                   position='top')

        frame = draw_metronome_bar(frame, current_beat, timeline.beats_per_bar,
                                   position='bottom')

        frame = draw_chord_timeline(frame, upcoming, current_bar,
                                    position='bottom-left')

        frame = draw_streak_counter(frame, streak, max_streak,
                                    position='top-right')

        # Draw current chord name prominently
        if current_chord:
            chord_display_y = h // 2 - 100
            overlay = frame.copy()
            cv2.rectangle(overlay, (w//2 - 100, chord_display_y - 40),
                         (w//2 + 100, chord_display_y + 10), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            chord_color = (0, 255, 0) if (score and score['accuracy'] >= 0.8) else (0, 165, 255)
            cv2.putText(frame, current_chord, (w//2 - 80, chord_display_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, chord_color, 4)

        # Show frame
        cv2.imshow("Play-Along", frame)

        # Draw volume indicator if music enabled
        if music_enabled:
            vol_text = f"Volume: {int(music_player.volume * 100)}%"
            cv2.putText(frame, vol_text, (w - 160, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show frame
        cv2.imshow("Play-Along", frame)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n⏹️  Stopped early")
            break
        elif key == ord('m') and music_enabled:
            # Toggle music
            if music_player.is_song_playing():
                music_player.pause()
            else:
                music_player.unpause()
        elif key == ord('=') or key == ord('+'):
            # Increase volume
            if music_enabled:
                music_player.increase_volume(0.1)
        elif key == ord('-') or key == ord('_'):
            # Decrease volume
            if music_enabled:
                music_player.decrease_volume(0.1)

    web_cam.release()
    cv2.destroyAllWindows()

    # Stop and cleanup music
    if music_enabled:
        music_player.cleanup()

    # Print session summary
    print("\n" + "="*60)
    print("📊 SESSION SUMMARY")
    print("="*60)
    print(f"Song: {song_data['title']} - {song_data['artist']}")
    print(f"Completed bars: {current_bar}/{total_bars}")
    print(f"Max streak: {max_streak} 🔥")

    if scores:
        avg_accuracy = sum(scores) / len(scores)
        print(f"Average accuracy: {avg_accuracy:.1%}")

        # Per-chord breakdown
        print("\nPer-chord accuracy:")
        for chord, accs in sorted(chord_scores.items()):
            avg = sum(accs) / len(accs)
            print(f"  {chord}: {avg:.1%} ({len(accs)} frames)")

    print("="*60 + "\n")

    return True
