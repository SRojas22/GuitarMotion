import sounddevice as sd
from scipy.io.wavfile import write
import librosa as lb
import numpy as np
import json
import time


def record_and_analyze_audio(duration=5):
    fs = 44100  # Sample rate
    print("Recording... ")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("Recording complete")

    # Save audio file
    write("test.wav", fs, recording)
    print("Saved audio as test.wav")

    # Load and analyze
    y, sr = lb.load("test.wav")
    print(f"Audio loaded successfully, samples: {len(y)}, sample rate: {sr}")

    # Detect pitch
    f0 = lb.yin(y, fmin=lb.note_to_hz('E2'), fmax=lb.note_to_hz('E6'))
    note_names = lb.hz_to_note(f0)
    print("Detected notes:")
    print(note_names[:20])

    # Timestamp and save to JSON
    session_data = []
    for i, note in enumerate(note_names):
        if note != 'nan':
            timestamp = i * (len(y)/len(f0)) / sr
            session_data.append({
                "time_s": round(timestamp, 3),
                "note": note
            })

    # Save JSON
    music_file = f"session_{int(time.time())}.json"
    with open(music_file, "w") as f:
        json.dump(session_data, f, indent=4)

    print(f"Session data saved to {music_file}")

    return music_file
