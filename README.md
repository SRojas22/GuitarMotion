# 🎸 GuitarMotion

AI-powered guitar learning tool that helps beginners learn guitar by tracking hand positions and comparing them with actual played notes.

## Features

- **Computer Vision Guitar Detection**: Automatically detects your physical guitar in the camera frame using edge detection and contour analysis
- **Calibration System**: User-friendly calibration phase to position and confirm guitar detection before recording
- **Real-time Hand Tracking**: Uses MediaPipe to detect your hand positions on the actual detected guitar fretboard
- **Note Prediction**: Predicts which notes you're trying to play based on finger positions relative to the detected guitar strings and frets
- **Audio Analysis**: Records and analyzes the actual notes you play using Librosa
- **Smart Comparison**: Compares predicted notes (from hand position on detected guitar) vs actual notes (from audio) to show if you're playing correctly
- **Visual Feedback**: Displays detected guitar boundaries, strings, frets, hand tracking, and predicted notes in real-time
- **Session Recording**: Saves detailed JSON logs of each practice session with accuracy metrics
- **Accuracy Tracking**: Shows your overall accuracy and identifies which notes you played correctly

## How It Works

1. **Guitar Detection**: Uses computer vision (Canny edge detection + contour analysis) to detect the physical guitar neck/fretboard in the camera frame
2. **Calibration**: User positions guitar and confirms detection with visual feedback (green box around detected guitar)
3. **String & Fret Mapping**: Automatically calculates positions of 6 strings and 12 frets on the detected guitar
4. **Hand Detection**: MediaPipe tracks your hand positions in real-time via webcam
5. **Coordinate Mapping**: Maps your index finger position to specific strings and frets on the DETECTED guitar (not just the frame)
6. **Note Prediction**: Determines which note you're trying to play based on standard tuning
7. **Audio Recording**: Simultaneously records the actual sound you make
8. **Audio Analysis**: Uses librosa's YIN algorithm to detect the actual notes from audio
9. **Comparison**: Matches predicted notes with actual notes and calculates accuracy
10. **Results**: Displays detailed feedback showing correct/incorrect notes

## Setup

1. Activate your virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Install dependencies (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Run the Program
```bash
python3 main.py
```

### Step 2: Guitar Calibration Phase
1. **Position your guitar**:
   - Hold guitar horizontally in the camera frame
   - Make sure the fretboard/neck is clearly visible
   - Ensure good lighting on the guitar
   - Guitar should take up about 1/3 to 1/2 of the frame

2. **Wait for detection**:
   - The program will automatically detect the guitar using computer vision
   - You'll see a yellow box showing the suggested position
   - When guitar is detected, a green box will appear around it

3. **Confirm calibration**:
   - Once you see "Guitar Detected!" message
   - Press **SPACE** to confirm and start the session
   - Press **'q'** to cancel and quit

### Step 3: Recording Session
1. After confirmation, you have 3 seconds to prepare
2. The camera window shows:
   - Detected guitar with string and fret overlays
   - Your hand tracking in real-time
   - Predicted notes as you place fingers on frets
   - Recording timer

3. **Play your guitar** - place fingers on strings/frets and play notes

4. Press **'q'** when done to stop and analyze

### Step 4: View Results
- Overall accuracy percentage
- Detailed comparison of each note (predicted vs actual)
- ✅ Green checkmarks for correct notes
- ❌ Red X marks for incorrect notes
- Session data automatically saved in `data/recordings/sessions/`

## Understanding the Results

After each session, you'll see:
- ✅ **Green checkmarks**: Notes you played correctly
- ❌ **Red X marks**: Notes that didn't match (wrong position or wrong sound)
- **Accuracy percentage**: How many notes you played correctly overall
- **Detailed log**: Time, predicted note, actual note, string, and fret for each attempt

## Requirements

- Python 3.10+
- Webcam
- Guitar
- Microphone (built-in or external)

## File Structure

```
GuitarMotion/
├── main.py                    # Main application with 3-phase workflow
├── src/
│   ├── guitar_detection.py    # Computer vision guitar detection
│   ├── hand_detection.py      # Hand tracking utilities (legacy)
│   └── audio_utils.py         # Audio processing utilities (legacy)
├── data/
│   └── recordings/
│       └── sessions/          # Session JSON files stored here
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Tips for Best Results

### For Guitar Detection:
- Use good, even lighting on the guitar fretboard
- Hold the guitar horizontally in the frame (neck going left-right)
- Position guitar so the fretboard/neck takes up 1/3 to 1/2 of the frame
- Avoid cluttered backgrounds - plain backgrounds work best
- If detection fails, try adjusting the angle or lighting
- Wooden/natural fretboards work better than very dark or reflective ones

### For Hand Tracking:
- Keep hands visible and well-lit
- Position hands clearly on the detected guitar fretboard
- Avoid rapid movements - smooth, deliberate finger placements work best
- Make sure fingers are clearly pressing specific frets

### For Audio Detection:
- Play in a quiet environment for better audio detection
- Hold notes for at least 0.5 seconds for better detection
- Play clearly - one note at a time works better than chords initially
- Ensure guitar is tuned to standard tuning (E A D G B E)

### General:
- Start with slow, simple notes to get familiar with the system
- Check the visual feedback - if predicted notes look wrong, recalibrate
- Review your session JSON files to see detailed tracking data

