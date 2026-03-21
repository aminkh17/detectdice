# Dice Detector Pro — Technical Documentation

`detector.py` is a real-time computer vision application designed to detect three specific colored dice (Red, Green, Blue), count their pips, and perform mathematical calculations based on the detected values.

## 1. Core Architecture

The script follows a modular pipeline:
1. **Frame Capture**: Pulls raw BGR frames from the webcam.
2. **Preprocessing**: Optional White Balance correction to stabilize colors.
3. **Color Segmentation**: Builds binary masks for each die color using HSV (Hue, Saturation, Value) ranges.
4. **Localization**: Identifies die-shaped contours within the masks.
5. **Feature Extraction**: Isolates the top face of the die and detects circular pips.
6. **Temporal Stability**: Filters out "flickering" by requiring consecutive consistent readings.
7. **UI & Calculation**: Overlays results and performs final formulas.

---

## 2. Real-Time Settings (Trackbars)

The application includes a "Settings" window with sliders to calibrate the detector for your environment:

| Slider | Description |
| :--- | :--- |
| **Sat Min** | **Saturation Minimum.** Filters out grey/black/white background objects. Increase this to ignore the table or speakers. |
| **Val Min** | **Value Minimum.** Filters out dark shadows. Increase this to ignore dark equipment. |
| **Red Range** | Adjusts how much of the Red spectrum is captured (0-180 wrap-around). |
| **Green Hue** | Centers the detection on your specific shade of green. |
| **Blue Hue** | Centers the detection on your specific shade of blue. |
| **Pip Circ** | **Circularity.** 1.0 is a perfect circle. Set to ~0.65 to count round pips but ignore dust or side-face glares. |
| **Face Focus** | Shrinks the "detection zone" toward the center of the die to avoid counting side-face pips. |
| **Area Min** | Minimum pixel area for an object to be considered a die. |
| **WB Multi** | Adjusts the strength of the automatic color correction. |

---

## 3. The Processing Pipeline

### A. Color Masking (`build_mask`)
The script converts BGR images to **HSV**.
- **Red** is unique as it exists at both the start (0-10) and end (170-180) of the hue spectrum.
- **Morphology**: It uses `MORPH_CLOSE` to fill small holes in the die and `MORPH_OPEN` to remove tiny noise particles.

### B. Dice Localization
It searches for contours in the cleaned-up masks. It filters candidates based on:
- **Area**: Is it big enough to be a die but small enough not to be a hand?
- **Squareness**: Is the Width/Height ratio roughly 1:1?

### C. Pip Counting (`count_pips`)
Once a die is found:
1. It crops the die area and converts it to **Grayscale**.
2. **Histogram Equalization**: Normalizes brightness so pips stand out against the die body.
3. **Otsu Thresholding**: Automatically finds the best contrast threshold to turn pips white and the die body black.
4. **Circularity Filter**: Uses the formula $4\pi 	imes \frac{Area}{Perimeter^2}$ to ensure only round pips are counted.

### D. Stability Tracking (`StabilityTracker`)
To prevent the numbers from jumping around due to camera noise, the script requires **10 consecutive frames** of the exact same pip count before it considers a die "Confirmed" (labeled as **OK** on screen).

---

## 4. Formulas & Results

Once all three dice (Red, Green, Blue) are confirmed, the script calculates:

1. **Index**:  
   $(R-1) \times 36 + (G-1) \times 6 + B$
2. **Final**:  
   $1 + \lfloor \frac{(Index-1) \times (TopNumber-1)}{215} \rfloor$ (default TopNumber = 35, i.e. $\times 34$)

---

## 5. Keyboard Shortcuts

- **`Q`**: Quit the application.
- **`D`**: **Debug Mode.** Overlays the internal "Mask" view and the "Pip Detection" view so you can see exactly what the computer sees.
- **`R`**: **Reset.** Clears all confirmed values and history buffers. Use this if you move the dice.

---

## 6. Requirements
- Python 3.8+
- `opencv-python`
- `numpy`
