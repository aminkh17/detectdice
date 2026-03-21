# DiceDetect

Real-time computer vision app that detects three colored dice (Red, Green, Blue) from a webcam, counts their pips, and computes derived Index / Final values.

Built with Python + OpenCV. No machine learning model required — pure HSV color segmentation + contour analysis + temporal stability filtering.

## Scripts

| File | Description |
| :--- | :--- |
| `detector.py` | Main app ("Dice Detector Pro"). Live trackbar settings window, white-balance correction, stability confirmation, Index/Final overlay. Supports camera index or IP/USB camera URL via `--camera`. |
| `dice_logic.py` | Pure math helpers: `calculate_index(r, g, b)` and `calculate_final(index, topnumber)`. No OpenCV dependency. |
| `recdetector.py` | Simpler fixed-threshold variant (no settings window). Good reference / fallback. Prints `R/G/B → Total` when all three dice are stable. |
| `DETECTOR_README.md` | Deep technical documentation of the detection pipeline. |

## Requirements

- Python 3.9+
- Webcam
- `opencv-python`, `numpy`

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python detector.py               # default camera 0
python detector.py --camera 1    # alternate camera
python detector.py --camera "http://192.168.1.10:8080/video"  # IP camera URL
```

Fallback detector:

```bash
python recdetector.py --camera 0
python recdetector.py --camera 0 --debug
```

## Usage

Point the camera at the three dice. Each die gets a colored bounding box:

- Thin box + `...` = tracking, not yet stable
- Thick box + `OK` / `✓` = confirmed (10 consecutive identical frames)

Once all three are confirmed, the app shows:

- **Index**: `(R-1) × 36 + (G-1) × 6 + B` → range 1–216
- **Final**: `1 + floor((Index-1) × (TopNumber-1) / 215)` → range 1–TopNumber (default 35)

### Keyboard shortcuts (`detector.py`)

- `Q` — quit
- `D` — toggle debug overlay (color masks + pip contours)
- `R` — reset confirmed values and history

### Keyboard shortcuts (`recdetector.py`)

- `Q` — quit, `R` — reset, `D` — debug overlay, `S` — save frame as `dice_capture_<timestamp>.png`

### Tuning (`detector.py` Settings window)

| Slider | Purpose |
| :--- | :--- |
| Sat Min / Val Min | Ignore grey background / dark shadows |
| Red Range / Green Hue / Blue Hue | Adapt to your dice + lighting |
| Pip Circ | Pip circularity threshold (~0.65) |
| Face Focus | Shrink detection zone to top face only |
| Area Min | Minimum die size in pixels |
| WB Multi | White-balance correction strength |
| Top Number | Upper bound for Final calculation |

## How it works

1. Optional white-balance correction (LAB space).
2. BGR → HSV, per-color threshold → binary mask, morphological close/open to denoise.
3. Contour search filtered by area + squareness → die bounding box.
4. Crop ROI → grayscale → blur → Otsu threshold → circularity-filtered pip contours.
5. `StabilityTracker` requires 10 consecutive identical readings before confirming.
6. Compute Index / Final and render overlay.

See `DETECTOR_README.md` for full pipeline details.

## Project structure

```text
.
├── detector.py          # main app
├── dice_logic.py        # index/final math (testable, no cv2)
├── recdetector.py       # simpler fixed-threshold variant
├── DETECTOR_README.md   # technical deep-dive
├── requirements.txt
└── images/              # put curated demo images here (images/demo-*.png)
```

Local captures (`dice_capture_*.png`), screenshots, and test `image*.png` files are git-ignored by design.

## License

MIT — see `LICENSE`.
