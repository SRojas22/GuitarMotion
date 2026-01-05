# 🎸 GuitarMotion - AI Guitar Learning Assistant

An AR guitar coach that locks onto your real guitar's fretboard, overlays strings/frets/chord shapes, and provides real-time feedback with play-along mode for your favorite songs.

![Version](https://img.shields.io/badge/version-2.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)

## Features

### Smart Fretboard Detection
- **Custom YOLOv8 Model**: Trained on 543+ images for 99.16% accuracy
- **Hybrid Detection**: YOLO + edge detection fallback for reliability
- **Smooth Lock-On**: EMA smoothing eliminates jitter with visual state transitions
- **Neon Glow UI**: Clear visual feedback (Searching → Locking → Locked)

### Precision Calibration
- **Two-Click Setup**: Simply click nut + 12th fret for perfect alignment
- **Real Guitar Math**: Logarithmic fret spacing (12th root of 2 formula)
- **20 Fret Support**: Full fretboard coverage with labeled fret numbers
- **Live Preview**: See string/fret overlay before you start

### Chord Coach Mode
- **14 Chords Library**: From beginner (C, G, Em) to advanced (Bm, Cadd9)
- **Ghost Circles**: Visual targets show exactly where to place fingers
- **Real-Time Scoring**: Instant accuracy feedback with color coding
- **Practice Tracking**: Session summaries with accuracy percentages

### Play-Along Mode
- **4 Songs Included**:
  - Horse with No Name (beginner, 122 BPM)
  - Let It Be (beginner, 73 BPM)
  - Wonderwall (beginner, 87 BPM)
  - Ojitos Lindos (intermediate, 79 BPM)
- **Background Music**: MP3 playback with volume control (M, +, -)
- **Visual Metronome**: Beat bars with downbeat accent
- **Chord Timeline**: See next 3 chords coming up
- **Streak Counter**: Track consecutive correct chords

### Recording Session Mode
- **Hand Tracking**: MediaPipe-based finger position detection
- **Audio Analysis**: Librosa pitch detection
- **Note Comparison**: Compare hand position vs actual sound
- **Session Logs**: Detailed JSON recordings for review

## Quick Start

### Prerequisites
- Python 3.10+
- Webcam
- Guitar
- Microphone (for Recording mode)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/GuitarMotion.git
cd GuitarMotion
```

2. **Create virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate 
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download the trained model**:
   - Place `fretboard_detector.pt` in `models/weights/`
   - Or train your own (see Training section below)

### Run the App

```bash
python3 main.py
```

## Usage Guide

### Phase 1: Fretboard Detection & Lock

1. **Position your guitar**:
   - Hold horizontally in frame
   - Fretboard clearly visible
   - Well-lit environment

2. **Wait for lock**:
   - Orange: Searching for fretboard
   - Yellow: Locking (detecting consistently)
   - Green: Locked 

3. **Confirm**: Press **SPACE** when locked

### Phase 2: Calibration

1. **Click the NUT** (leftmost edge, 0th fret)
2. **Click the 12TH FRET**
3. **Verify**: Check fret labels align with your guitar
4. **Continue**: Press **SPACE** to confirm

### Phase 3: Choose Your Mode

#### Chord Coach (Mode 2)
```
1. Select a chord (C, G, D, Em, Am, etc.)
2. See ghost circles showing target positions
3. Place your fingers - get instant feedback
4. Green = correct, Orange = close, Red = wrong
5. Press 'c' to change chord, 'q' to quit
```

**Best for**: Learning new chords, improving accuracy

#### Play-Along (Mode 3)
```
1. Choose a song (1-4)
2. Background music starts automatically
3. Follow chord changes on screen
4. Build streaks for consecutive correct chords
5. Controls: M (pause/play), + (volume up), - (volume down)
```

**Best for**: Practicing chord changes, playing with music

#### Recording Session (Mode 1)
```
1. Start recording (3 second countdown)
2. Play notes/chords
3. Press 'q' to stop
4. See accuracy report (hand position vs actual sound)
```

**Best for**: Analyzing your technique, detailed feedback



**Controls**:
- **SPACE**: Capture image
- **L**: Toggle lighting (bright/normal/dim)
- **A**: Toggle angle (straight/tilt-left/tilt-right)
- **G**: Toggle guitar type (acoustic/classical/electric)
- **B**: Toggle background (plain/cluttered)
- **Q**: Quit

**Target**: 300-600 images with varied conditions

### Labeling

Use [Roboflow](https://roboflow.com) or LabelImg:
1. Upload images to Roboflow
2. Draw bounding boxes around fretboard (class: `fretboard`)
3. Export in YOLOv8 format
4. Place in `data/labeled/`

### Training

```bash
python3 models/train_fretboard_detector.py
```

Model saved to: `models/weights/fretboard_detector.pt`

## Project Structure

```
GuitarMotion/
├── main.py                          # Main application entry point
├── requirements.txt                 # Python dependencies
│
├── models/
│   ├── fretboard_detector.py        # YOLO wrapper with EMA smoothing
│   ├── train_fretboard_detector.py  # Training script
│   └── weights/
│       └── fretboard_detector.pt    # Trained model (not in git)
│
├── vision/
│   ├── hybrid_detector.py           # YOLO + edge fallback
│   ├── lock_state.py                # State machine (SEARCHING/LOCKING/LOCKED)
│   └── fretboard_mapper.py          # String/fret geometry + calibration
│
├── modes/
│   ├── chord_coach.py               # Chord practice logic
│   ├── chord_coach_session.py       # Chord coach UI loop
│   ├── play_along_session.py        # Play-along UI loop
│   └── song_timeline.py             # BPM-based timeline manager
│
├── ui/
│   ├── overlays.py                  # Lock state, glow, instructions
│   ├── calibration_ui.py            # Two-click calibration
│   └── metronome.py                 # Beat bar, chord timeline, streak
│
├── audio/
│   └── background_music.py          # Pygame music player
│
├── tools/
│   ├── collect_fretboard_data.py    # Image capture tool
│   └── organize_labeled_data.py     # LabelImg organizer
│
├── data/
│   ├── chords.json                  # 14 chord library
│   ├── songs/                       # Song JSONs + audio
│   │   ├── wonderwall.json
│   │   ├── ojitos_lindos.json
│   │   └── audio/                   # MP3 files (not in git)
│   └── recordings/
│       └── sessions/                # Session logs (not in git)
│
└── src/
    └── guitar_detection.py          # Edge-based fallback detector
```

### Calibration Math
- Fret spacing: `position = nut + (scale - scale / 2^(fret/12))`
- Scale length: 2 × (12th fret - nut)
- String spacing: Even distribution across bbox height

### Hand Tracking
- MediaPipe Hands (1 hand max)
- Fingertip landmarks: 8 (index), 12 (middle), 16 (ring), 20 (pinky)
- Mapped to nearest string/fret intersection

## Areas to improve:
this are things tath i still need to work on to make it more usable:
- More chords (power chords, 7ths, etc.)
- Additional songs
- Strumming pattern detection
- Tempo adjustment in play-along
- Mobile app version

## Acknowledgments

- **YOLOv8**: Ultralytics
- **Hand Tracking**: Google MediaPipe
- **Audio Analysis**: Librosa
- **Computer Vision**: OpenCV

