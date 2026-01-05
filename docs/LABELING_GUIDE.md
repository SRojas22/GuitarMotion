# Fretboard Labeling Guide

This guide will help you label the fretboard images you collected for YOLOv8 training.

## Install LabelImg

```bash
pip install labelImg
```

## Labeling Workflow

### 1. Launch LabelImg

```bash
labelImg
```

### 2. Configure LabelImg for YOLO Format

1. **Change Save Format**:
   - Click `Edit` > `Change Save Format`
   - Select `YOLO`

2. **Set Image Directory**:
   - Click `Open Dir`
   - Navigate to your session folder (e.g., `data/raw_images/session_1234567890`)

3. **Set Save Directory**:
   - Click `Change Save Dir`
   - Create/select: `data/labeled/labels/train` (for training set)
   - Later you'll also use `val` and `test` subdirectories

### 3. Label Each Image

For each image:

1. Press `W` or click "Create RectBox"
2. **Draw a tight bounding box around the fretboard**:
   - **Start point**: At the nut (where the headstock meets the fretboard)
   - **End point**: At the visible end of the neck/fretboard
   - **Include**: All fret markers, fret wires, strings
   - **Exclude**: Guitar body, headstock

3. **Label as "fretboard"** (class 0)
   - When prompted, type: `fretboard`
   - This will be class index 0 in your dataset

4. Press `D` to go to next image
5. Repeat for all images

## Labeling Rules

### ✅ Correct Labeling

- **Tight fit**: Box should closely fit the fretboard region
- **Nut to end**: From the nut line to the visible end of the neck
- **Include fret markers**: Dots/inlays should be inside the box
- **Partial occlusion OK**: If hands partially cover the fretboard, still label the entire fretboard region

### ❌ Avoid These Mistakes

- **Too loose**: Don't include excessive background
- **Including body**: Guitar body should be outside the box
- **Including headstock**: Headstock should be outside the box
- **Missing edges**: Don't cut off the fretboard edges

## Visual Examples

```
GOOD: Tight box around fretboard
┌─────────────────────────────────┐
│nut    frets    markers      end│  ← Fretboard only
└─────────────────────────────────┘

BAD: Too much background
    ┌────────────────────────────────────┐
    │        fretboard                   │  ← Includes body/background
    └────────────────────────────────────┘

BAD: Missing part of fretboard
         ┌──────────────────┐
  nut────│   fretboard      │──end  ← Nut is outside
         └──────────────────┘
```

## Dataset Split Strategy

Label images in batches and organize into train/val/test:

- **Training set (80%)**: ~240 images → save labels to `data/labeled/labels/train/`
- **Validation set (10%)**: ~30 images → save labels to `data/labeled/labels/val/`
- **Test set (10%)**: ~30 images → save labels to `data/labeled/labels/test/`

### Recommended Approach

1. **Label all images first**, saving to `train` directory
2. **Then run the organization script** (`tools/organize_labeled_data.py`) to automatically split them

OR

3. **Manual split**: As you label, periodically change the save directory to `val` or `test`

## Tips for Efficient Labeling

1. **Keyboard shortcuts**:
   - `W` - Create box
   - `D` - Next image
   - `A` - Previous image
   - `Ctrl+S` - Save
   - `Del` - Delete selected box

2. **Zoom in** on edges if needed for precise bounding boxes

3. **Take breaks** every 50-100 images to avoid fatigue

4. **Check variety**: Ensure you're labeling images with different:
   - Lighting conditions
   - Angles
   - Guitar types
   - Backgrounds

## Quality Check

Before moving to training:

1. **Spot check**: Review ~20 random labeled images
2. **Verify YOLO format**: Check that `.txt` files exist next to images
3. **Check class index**: Each `.txt` file should start with `0` (class fretboard)

YOLO format example (`img_0001.txt`):
```
0 0.5 0.45 0.65 0.25
```
Format: `class_id x_center y_center width height` (all normalized 0-1)

## Next Steps

After labeling:

1. ✅ Run `tools/organize_labeled_data.py` to split dataset
2. ✅ Verify `data/data.yaml` was created
3. ✅ Run `models/train_fretboard_detector.py` to train the model

## Troubleshooting

**Q: LabelImg won't save in YOLO format?**
- A: Make sure you selected YOLO format via `Edit` > `Change Save Format` > `YOLO`

**Q: Can't find my images?**
- A: Use `Open Dir` to navigate to `data/raw_images/session_XXXXX/`

**Q: How tight should the box be?**
- A: Aim for 2-5 pixels of padding around the fretboard edges. Too tight can miss edges, too loose includes irrelevant background.

**Q: Should I label if the fretboard is partially out of frame?**
- A: Yes! Label the visible portion only. This helps the model learn to detect partial fretboards.

**Q: What if the image is blurry or bad quality?**
- A: Skip it (don't label). Bad training images hurt model performance.
