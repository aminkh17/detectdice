"""
dice_detector.py
────────────────
Detects red, green and blue dice via webcam using OpenCV.
Returns the face value of each die and the running total.

Requirements:
    pip install opencv-python numpy

Usage:
    python dice_detector.py              # default camera 0
    python dice_detector.py --camera 1   # alternate camera index
    python dice_detector.py --debug      # show pip-detection overlay

Keyboard shortcuts (while window is open):
    Q – quit
    R – reset all readings
    D – toggle debug overlay
    S – save current frame as PNG
"""

from __future__ import annotations   # ← fixes int | None on Python < 3.10

import argparse
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ColourProfile:
    name: str
    hsv_ranges: List[Tuple[np.ndarray, np.ndarray]]   # list of (low, high)
    bgr: Tuple[int, int, int]                          # box / label colour
    prefix: str                                        # single-letter prefix


@dataclass
class DiceReading:
    colour: str
    pips: int
    bbox: Tuple[int, int, int, int]   # x, y, w, h
    confirmed: bool = False


# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR PROFILES  (tweak HSV ranges for your lighting)
# ══════════════════════════════════════════════════════════════════════════════

COLOUR_PROFILES: Dict[str, ColourProfile] = {
    "Red": ColourProfile(
        name="Red",
        hsv_ranges=[
            (np.array([0,   70,  50], np.uint8), np.array([15,  255, 255], np.uint8)),
            (np.array([165, 70,  50], np.uint8), np.array([180, 255, 255], np.uint8)),
        ],
        bgr=(0, 0, 220),
        prefix="R",
    ),
    "Green": ColourProfile(
        name="Green",
        hsv_ranges=[
            (np.array([40,  60,  60], np.uint8), np.array([90,  255, 255], np.uint8)),
        ],
        bgr=(0, 200, 0),
        prefix="G",
    ),
    "Blue": ColourProfile(
        name="Blue",
        hsv_ranges=[
            (np.array([95,  80,  50], np.uint8), np.array([135, 255, 255], np.uint8)),
        ],
        bgr=(220, 60, 0),
        prefix="B",
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
#  TUNABLE CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

MIN_DICE_AREA        = 1_200    # px²  – smaller blobs ignored
MAX_DICE_AREA        = 80_000   # px²  – larger blobs (hands) ignored
MIN_DOT_AREA         = 18       # px²  – minimum pip size
MAX_DOT_AREA         = 1_500    # px²  – maximum pip size
MIN_DOT_CIRCULARITY  = 0.55     # 0–1, 1 = perfect circle
STABLE_FRAMES        = 10       # consecutive identical frames to confirm
MORPH_KERNEL_SIZE    = 5        # morphology kernel (odd number)
ASPECT_RATIO_MIN     = 0.55     # die bounding box aspect ratio limits
ASPECT_RATIO_MAX     = 1.82
ROI_PAD              = 6        # pixels to expand each detected die region


# ══════════════════════════════════════════════════════════════════════════════
#  MASK BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_colour_mask(hsv: np.ndarray, profile: ColourProfile) -> np.ndarray:
    """
    Combine all HSV sub-ranges for a colour into one clean binary mask.
    Morphological close+open removes noise and fills small gaps.
    """
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in profile.hsv_ranges:
        combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lo, hi))

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel, iterations=1)
    return combined


# ══════════════════════════════════════════════════════════════════════════════
#  DICE CONTOUR FINDER
# ══════════════════════════════════════════════════════════════════════════════

def find_dice_boxes(
    mask: np.ndarray,
) -> List[Tuple[int, int, int, int]]:
    """
    Find bounding boxes of die-shaped blobs in a binary mask.
    Returns list of (x, y, w, h) sorted by area descending.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_DICE_AREA < area < MAX_DICE_AREA):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h)
        if ASPECT_RATIO_MIN < aspect < ASPECT_RATIO_MAX:
            boxes.append((x, y, w, h, area))

    # sort largest first so dominant die wins if two overlap
    boxes.sort(key=lambda b: b[4], reverse=True)
    return [(x, y, w, h) for x, y, w, h, _ in boxes]


# ══════════════════════════════════════════════════════════════════════════════
#  PIP COUNTER
# ══════════════════════════════════════════════════════════════════════════════

def count_pips(
    roi_gray: np.ndarray,
    debug: bool = False,
) -> Tuple[int, Optional[np.ndarray]]:
    """
    Count white circular pips on a die face (greyscale ROI).

    Returns
    -------
    (pip_count, debug_img)
        pip_count  : integer 0-6  (0 means detection failed)
        debug_img  : colour image with pip contours drawn, or None
    """
    # ── Guard: skip if ROI is too small to be valid ───────────────────────
    if roi_gray is None or roi_gray.size == 0:
        return 0, None
    if roi_gray.shape[0] < 10 or roi_gray.shape[1] < 10:
        return 0, None

    blur   = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Pips are lighter than the die face; if majority is white it means the
    # background thresholded white → invert so pips become white blobs.
    white_ratio = np.count_nonzero(thresh) / thresh.size
    if white_ratio > 0.5:
        thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    debug_img   = None
    pip_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_DOT_AREA < area < MAX_DOT_AREA):
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity >= MIN_DOT_CIRCULARITY:
            pip_contours.append(cnt)

    pip_count = min(len(pip_contours), 6)

    # ── Optional debug overlay ────────────────────────────────────────────
    if debug:
        debug_img = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_img, pip_contours, -1, (0, 255, 255), 1)
        for cnt in pip_contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(debug_img, (cx, cy), 3, (0, 0, 255), -1)

    return pip_count, debug_img


# ══════════════════════════════════════════════════════════════════════════════
#  STABILITY TRACKER  (per colour+position bucket)
# ══════════════════════════════════════════════════════════════════════════════

class StabilityTracker:
    """
    Confirms a pip reading only after it has been STABLE_FRAMES consecutive
    identical readings.  Tracks per colour (one die per colour expected).
    """

    def __init__(self, required: int = STABLE_FRAMES) -> None:
        self.required  = required
        self._history: Dict[str, deque] = {}

    def update(self, colour: str, value: int) -> Optional[int]:
        """
        Push a new reading.  Returns confirmed value when stable, else None.
        """
        if colour not in self._history:
            self._history[colour] = deque(maxlen=self.required)
        self._history[colour].append(value)

        buf = self._history[colour]
        if len(buf) == self.required and len(set(buf)) == 1:
            return buf[0]
        return None

    def clear(self, colour: str) -> None:
        self._history.pop(colour, None)

    def reset(self) -> None:
        self._history.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def draw_label(
    img: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    bg_colour: Tuple[int, int, int],
    font_scale: float = 0.65,
    thickness: int = 2,
) -> None:
    """Draw a filled-background text label at *origin* (top-left of box)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    # background rectangle
    cv2.rectangle(img, (x, y - th - baseline - 4), (x + tw + 4, y),
                  bg_colour, cv2.FILLED)
    # white text
    cv2.putText(img, text, (x + 2, y - baseline - 2),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_status_bar(
    img: np.ndarray,
    confirmed: Dict[str, int],
    fps: float,
    debug_on: bool,
) -> None:
    """Render the top-left status overlay."""
    all_found = len(confirmed) == 3
    colour    = (0, 210, 0) if all_found else (0, 0, 220)

    if all_found:
        total = sum(confirmed.values())
        line1 = (f"R:{confirmed['Red']}  "
                 f"G:{confirmed['Green']}  "
                 f"B:{confirmed['Blue']}  "
                 f"→ Total={total}")
    else:
        parts = []
        for name in ("Red", "Green", "Blue"):
            parts.append(f"{name[0]}:{'✓' if name in confirmed else '?'}")
        line1 = "Waiting...  " + "  ".join(parts)

    cv2.putText(img, line1, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2, cv2.LINE_AA)

    info = f"FPS:{fps:4.1f}  {'[DEBUG]' if debug_on else ''}"
    cv2.putText(img, info, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    hint = "Q=Quit  R=Reset  D=Debug  S=Save"
    cv2.putText(img, hint,
                (10, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 150, 150), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main(camera_index: int = 0, debug: bool = False) -> None:
    # ── Open camera ───────────────────────────────────────────────────────
    # CAP_DSHOW is Windows-only and breaks camera open on macOS/Linux,
    # so only use it on Windows and fall back to the default backend.
    import platform
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        # fallback without backend hint
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,      1)    # request autofocus
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,  1)    # request auto-exposure

    tracker   = StabilityTracker(STABLE_FRAMES)
    confirmed: Dict[str, int] = {}
    last_print: Dict[str, int] = {}             # avoid terminal spam

    # FPS calculation
    fps_counter = 0
    fps_timer   = time.time()
    fps_display = 0.0

    print(__doc__)
    print("─" * 60)
    print(f"Camera {camera_index} opened. Waiting for dice…")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame capture failed – retrying…")
            time.sleep(0.05)
            continue

        display = frame.copy()
        hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        grey    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        seen_colours: set = set()

        # ── Per-colour detection ──────────────────────────────────────────
        for colour_name, profile in COLOUR_PROFILES.items():
            mask = build_colour_mask(hsv, profile)
            boxes = find_dice_boxes(mask)

            if not boxes:
                tracker.clear(colour_name)
                confirmed.pop(colour_name, None)
                continue

            # Take the largest/best candidate die for this colour
            x, y, w, h = boxes[0]
            rx1 = max(x - ROI_PAD, 0)
            ry1 = max(y - ROI_PAD, 0)
            rx2 = min(x + w + ROI_PAD, frame.shape[1])
            ry2 = min(y + h + ROI_PAD, frame.shape[0])

            roi_gray = grey[ry1:ry2, rx1:rx2]
            pips, dbg_img = count_pips(roi_gray, debug=debug)

            # Only track valid readings (1-6)
            if pips < 1:
                tracker.clear(colour_name)
                continue

            seen_colours.add(colour_name)
            stable_val = tracker.update(colour_name, pips)
            if stable_val is not None:
                confirmed[colour_name] = stable_val

            display_pips = confirmed.get(colour_name, pips)

            # ── Draw bounding box ─────────────────────────────────────────
            box_clr = profile.bgr
            thickness = 3 if colour_name in confirmed else 1
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), box_clr, thickness)

            label = f"{profile.prefix}:{display_pips}"
            if colour_name in confirmed:
                label += " ✓"
            draw_label(display, label, (rx1, ry1), box_clr)

            # ── Debug pip overlay ─────────────────────────────────────────
            if debug and dbg_img is not None:
                dh, dw = dbg_img.shape[:2]
                # clamp so it fits on screen
                dh = min(dh, display.shape[0] - ry1)
                dw = min(dw, display.shape[1] - rx1)
                display[ry1:ry1+dh, rx1:rx1+dw] = dbg_img[:dh, :dw]

        # Clear tracker for colours not seen this frame
        for colour_name in list(COLOUR_PROFILES.keys()):
            if colour_name not in seen_colours:
                tracker.clear(colour_name)
                confirmed.pop(colour_name, None)

        # ── Print to terminal only when result CHANGES ────────────────────
        if len(confirmed) == 3 and confirmed != last_print:
            total = sum(confirmed.values())
            print(
                f"[RESULT] Red={confirmed['Red']}  "
                f"Green={confirmed['Green']}  "
                f"Blue={confirmed['Blue']}  "
                f"Total={total}"
            )
            last_print = confirmed.copy()

        # ── FPS ───────────────────────────────────────────────────────────
        fps_counter += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_display  = fps_counter / elapsed
            fps_counter  = 0
            fps_timer    = time.time()

        draw_status_bar(display, confirmed, fps_display, debug)

        cv2.imshow("Dice Detector", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            confirmed.clear()
            last_print.clear()
            tracker.reset()
            print("[RESET] All readings cleared.")
        elif key == ord('d'):
            debug = not debug
            print(f"[DEBUG] {'ON' if debug else 'OFF'}")
        elif key == ord('s'):
            filename = Path(f"dice_capture_{int(time.time())}.png")
            cv2.imwrite(str(filename), display)
            print(f"[SAVED] {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("Exited cleanly.")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-colour dice detector via webcam.")
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Show pip-detection overlay on each die",
    )
    args = parser.parse_args()
    main(camera_index=args.camera, debug=args.debug)