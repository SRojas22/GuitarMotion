# Data Collection Strategy for High-Quality Training

This guide will help you collect the best possible training data for your fretboard detector.

## 🎯 Goal: 300-600 High-Quality, Varied Images

**Minimum viable**: 200-300 images (will work, but limited)
**Recommended**: 300-500 images (good performance)
**Excellent**: 500-1000+ images (robust in many conditions)

---

## 📋 Pre-Collection Checklist

### Setup
- [ ] Camera works and has good resolution (720p or better)
- [ ] Good lighting available (can adjust brightness)
- [ ] Guitar is accessible and tuned (optional but good practice)
- [ ] Clear space where you can move guitar around
- [ ] 20-40 minutes of uninterrupted time

### Install Dependencies (if not done)
```bash
cd /Users/susanarojas/Desktop/GuitarMotion
pip install -r requirements.txt
```

---

## 🎸 Collection Sessions - Best Practices

### Session 1: Foundation Set (100-150 images)
**Goal**: Cover basic positions in good lighting

**Setup**:
- Good, even lighting (daylight or bright room light)
- Plain background (white wall, solid color)
- One guitar (your main practice guitar)

**What to capture**:
1. **Straight-on views** (50 images):
   - Guitar horizontal, fretboard facing camera
   - Vary distance (close, medium, far)
   - Move guitar left/right in frame
   - Keep fretboard fully visible

2. **Different fret ranges** (50 images):
   - Show frets 1-5 (close to nut)
   - Show frets 5-12 (middle of neck)
   - Show frets 1-12 (full neck if possible)

3. **With hands** (30 images):
   - Hand positioning on frets
   - Different chord shapes
   - Both hands visible
   - Partial occlusion of fretboard is OK

**Keyboard shortcuts to use**:
- Keep: `[L] Lighting: normal`, `[A] Angle: straight`, `[B] Background: plain`
- Change `[G]` guitar type if you have multiple guitars

---

### Session 2: Lighting Variations (80-100 images)
**Goal**: Train model to work in different lighting

**Capture same positions as Session 1, but with**:

1. **Bright lighting** (40 images):
   - Next to window with daylight
   - Multiple bright lamps
   - Press `[L]` to set "bright"

2. **Dim lighting** (40 images):
   - Evening/night with soft lamp
   - Lower room lighting
   - Press `[L]` to set "dim"

**Tip**: Don't make it too dark - fretboard should still be visible to camera

---

### Session 3: Angle Variations (80-100 images)
**Goal**: Handle different camera angles

**Setup**: Back to normal lighting

1. **Tilt left** (25 images):
   - Tilt guitar 15-30° to the left
   - Press `[A]` to cycle to "tilt-left"

2. **Tilt right** (25 images):
   - Tilt guitar 15-30° to the right
   - Press `[A]` to cycle to "tilt-right"

3. **Top-down view** (25 images):
   - Camera looking down at fretboard at ~45° angle
   - As if you're looking at your own guitar while playing
   - Press `[A]` to cycle to "top-down"

4. **Mixed angles** (25 images):
   - Slight rotations
   - Camera slightly above/below
   - Fretboard at diagonal

---

### Session 4: Background Variations (50-80 images)
**Goal**: Work in real-world cluttered environments

**Capture with**:
1. **Cluttered background** (40 images):
   - Books, furniture, posters in background
   - Busy room setting
   - Press `[B]` to toggle "cluttered"

2. **Different backgrounds** (40 images):
   - Wood wall/table
   - Patterned background
   - Multiple colors

**Why**: Teaches model to distinguish fretboard from complex scenes

---

### Session 5: Edge Cases & Hard Examples (40-60 images)
**Goal**: Train model to handle challenging scenarios

1. **Partial fretboard** (15 images):
   - Only show half the fretboard (cut off by frame edge)
   - Fretboard partially out of frame

2. **Extreme angles** (15 images):
   - Very tilted positions
   - Camera at unusual height

3. **Hands heavily occluding** (15 images):
   - Fingers covering fret markers
   - Both hands positioned on frets

4. **Distance variations** (15 images):
   - Very close (fretboard fills frame)
   - Very far (fretboard is small in frame)

---

## 🔑 Quality Over Quantity

### ✅ Good Images Have:
- Fretboard clearly visible (not blurry)
- Good exposure (not too bright/dark)
- Fretboard in focus
- Variety in position within frame

### ❌ Skip Images That Are:
- Completely blurry/out of focus
- Overexposed (washed out/white)
- Underexposed (too dark to see frets)
- Fretboard barely visible (< 10% of frame)

---

## 📊 Recommended Image Distribution

| Category | Target | Notes |
|----------|--------|-------|
| Straight, good lighting, plain bg | 100 | Foundation |
| Bright lighting | 40 | Windows, bright lamps |
| Dim lighting | 40 | Evening, soft light |
| Tilted angles | 50 | Left, right, diagonal |
| Top-down view | 30 | Player's perspective |
| Cluttered background | 40 | Real environment |
| With hands | 50 | Practical use case |
| Edge cases | 50 | Partial, extreme angles |
| **TOTAL** | **400** | **Excellent dataset** |

---

## 🚀 Quick Start: Running the Collection Tool

```bash
cd /Users/susanarojas/Desktop/GuitarMotion
python tools/collect_fretboard_data.py
```

### While Collecting:

1. **Position guitar** in yellow suggestion box
2. **Toggle labels** as you change conditions:
   - `L` - Lighting (bright/normal/dim)
   - `A` - Angle (straight/tilt-left/tilt-right/top-down)
   - `G` - Guitar type (acoustic/classical/electric)
   - `B` - Background (plain/cluttered)
3. **Press SPACE** to capture each image
4. **Watch progress bar** toward 300/600 target
5. **Press Q** when done to save session

### After Each Session:

- Review the session summary
- Check variety breakdown
- Plan what to vary in next session

---

## 💡 Pro Tips

### 1. **Multiple Short Sessions > One Long Session**
- Take breaks between sessions
- Adjust your setup between sessions
- Better variety

### 2. **Move the Guitar, Not Just the Labels**
- Physically move guitar around
- Change distance from camera
- Rotate, tilt, shift position

### 3. **Capture "In-Between" States**
- Not just perfectly straight
- Not just perfectly centered
- Real-world positions

### 4. **Use Different Guitars (if available)**
- Acoustic vs classical vs electric
- Different neck widths
- Different fretboard colors (light vs dark wood)

### 5. **Test Your Camera First**
- Make sure webcam is working
- Check if image quality is good
- Adjust room lighting if needed

---

## ✅ Collection Checklist

Before moving to labeling, verify you have:

- [ ] At least 300 images total
- [ ] Good variety in lighting (not all same brightness)
- [ ] Good variety in angles (not all straight-on)
- [ ] Mix of plain and cluttered backgrounds
- [ ] Some images with hands visible
- [ ] All images have fretboard clearly visible
- [ ] Session metadata.json files created

---

## 🎯 Ready to Start!

Run this now:
```bash
python tools/collect_fretboard_data.py
```

**Suggested first session**:
- 30 minutes
- Normal room lighting
- Plain background
- Straight angles
- Just move guitar around and press SPACE
- Goal: 100-150 images

After you finish your first session, you can always run the tool again for more sessions!
