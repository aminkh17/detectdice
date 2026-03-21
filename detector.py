import cv2
import numpy as np
import argparse
from collections import deque
from dice_logic import calculate_index, calculate_final

# ==============================
# 1. Core Architecture
# ==============================

# ==============================
# 2. Real-Time Settings (Trackbars)
# ==============================
def nothing(x): pass

def setup_settings_window():
    cv2.namedWindow("Settings", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Settings", 450, 650)
    
    # Saturation is key to ignoring the grey speaker/table
    cv2.createTrackbar("Sat Min", "Settings", 130, 255, nothing)
    cv2.createTrackbar("Val Min", "Settings", 60, 255, nothing)
    
    # Hues - Tighter ranges to prevent color bleeding
    cv2.createTrackbar("Red Range", "Settings", 10, 20, nothing)   # ± from 0/180
    cv2.createTrackbar("Green Hue", "Settings", 75, 180, nothing)  # Leafy green center
    cv2.createTrackbar("Blue Hue", "Settings", 110, 180, nothing)  # Blue center
    
    # Pip/Detection Settings
    cv2.createTrackbar("Pip Circ", "Settings", 65, 100, nothing) 
    cv2.createTrackbar("Face Focus", "Settings", 75, 100, nothing)
    cv2.createTrackbar("Area Min", "Settings", 1000, 5000, nothing)
    cv2.createTrackbar("WB Multi x10", "Settings", 11, 30, nothing)
    
    # Logic Settings
    cv2.createTrackbar("Top Number", "Settings", 35, 100, nothing)

def get_settings():
    return {
        "sat_min": cv2.getTrackbarPos("Sat Min", "Settings"),
        "val_min": cv2.getTrackbarPos("Val Min", "Settings"),
        "red_range": cv2.getTrackbarPos("Red Range", "Settings"),
        "green_h": cv2.getTrackbarPos("Green Hue", "Settings"),
        "blue_h": cv2.getTrackbarPos("Blue Hue", "Settings"),
        "circularity": cv2.getTrackbarPos("Pip Circ", "Settings") / 100.0,
        "face_focus": cv2.getTrackbarPos("Face Focus", "Settings") / 100.0,
        "area_min": cv2.getTrackbarPos("Area Min", "Settings"),
        "wb_multi": cv2.getTrackbarPos("WB Multi x10", "Settings") / 10.0,
        "top_number": cv2.getTrackbarPos("Top Number", "Settings")
    }

# ==============================
# 3. The Processing Pipeline
# ==============================

def apply_white_balance(frame, multiplier):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    avg_a, avg_b = np.average(lab[:, :, 1]), np.average(lab[:, :, 2])
    l_mask = lab[:, :, 0] / 255.0
    lab[:, :, 1] -= (avg_a - 128) * l_mask * multiplier
    lab[:, :, 2] -= (avg_b - 128) * l_mask * multiplier
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

# 3.A. Color Masking
def build_mask(hsv, color_name, s):
    sat_min = s["sat_min"]
    val_min = s["val_min"]
    
    if color_name == "Red":
        # Red spans both ends of HSV
        r_range = s["red_range"]
        mask1 = cv2.inRange(hsv, np.array([0, sat_min, val_min]), np.array([r_range, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([180 - r_range, sat_min, val_min]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
    elif color_name == "Green":
        h = s["green_h"]
        mask = cv2.inRange(hsv, np.array([max(0, h-15), sat_min, val_min]), np.array([min(180, h+15), 255, 255]))
    else: # Blue
        h = s["blue_h"]
        mask = cv2.inRange(hsv, np.array([max(0, h-15), sat_min, val_min]), np.array([min(180, h+15), 255, 255]))
    
    # Cleaner morphology to avoid merging noise into "blobs"
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

# 3.C. Pip Counting
def count_pips(roi_gray, min_circ, face_focus, debug=False):
    if roi_gray is None or roi_gray.size < 100: return 0, None
    blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(thresh) / thresh.size > 0.5: thresh = cv2.bitwise_not(thresh)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pip_count = 0
    h, w = roi_gray.shape
    center_x, center_y = w // 2, h // 2
    max_dist = min(w, h) * 0.5 * face_focus 

    debug_img = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR) if debug else None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (w*h*0.005) < area < (w*h*0.1):
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            if np.sqrt((cx-center_x)**2 + (cy-center_y)**2) < max_dist:
                peri = cv2.arcLength(cnt, True)
                if peri > 0 and (4 * np.pi * (area / (peri * peri))) > min_circ:
                    pip_count += 1
                    if debug: cv2.drawContours(debug_img, [cnt], -1, (0, 255, 255), 1)

    return min(pip_count, 6), debug_img

# 3.D. Stability Tracking
class StabilityTracker:
    def __init__(self, required=10):
        self.required = required
        self.history = {}
    def update(self, name, val):
        if name not in self.history: self.history[name] = deque(maxlen=self.required)
        self.history[name].append(val)
        if len(self.history[name]) == self.required and len(set(self.history[name])) == 1:
            return self.history[name][0]
        return None
    def clear(self, name): self.history.pop(name, None)

def main(camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    setup_settings_window()
    tracker = StabilityTracker(required=10)
    confirmed = {}
    debug_mode = False

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        s = get_settings()
        corrected = apply_white_balance(frame, s["wb_multi"])
        hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
        
        current_frame_results = {}

        if debug_mode:
            m_r = cv2.cvtColor(build_mask(hsv, "Red", s), cv2.COLOR_GRAY2BGR)
            m_g = cv2.cvtColor(build_mask(hsv, "Green", s), cv2.COLOR_GRAY2BGR)
            m_b = cv2.cvtColor(build_mask(hsv, "Blue", s), cv2.COLOR_GRAY2BGR)
            cv2.imshow("MASK DEBUG (R-G-B)", np.vstack([m_r, m_g, m_b]))

        for color in ["Red", "Green", "Blue"]:
            mask = build_mask(hsv, color, s)
            
            # 3.B. Dice Localization
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_die = None
            max_area = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if s["area_min"] < area < 15000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if 0.7 < (w/h) < 1.4: # Stricter squareness
                        if area > max_area:
                            max_area = area
                            best_die = (x, y, w, h)
            
            if best_die:
                x, y, w, h = best_die
                y1, y2 = max(0, y-5), min(frame.shape[0], y+h+5)
                x1, x2 = max(0, x-5), min(frame.shape[1], x+w+5)
                roi = gray[y1:y2, x1:x2]
                
                pips, dbg_img = count_pips(roi, s["circularity"], s["face_focus"], debug_mode)
                
                if pips > 0:
                    stable = tracker.update(color, pips)
                    if stable: confirmed[color] = stable
                    current_frame_results[color] = {"val": confirmed.get(color, pips), "box": best_die}
                    
                    if debug_mode and dbg_img is not None:
                        dh, dw = dbg_img.shape[:2]
                        dy, dx = y1, x1
                        dh, dw = min(dh, frame.shape[0]-dy), min(dw, frame.shape[1]-dx)
                        frame[dy:dy+dh, dx:dx+dw] = dbg_img[:dh, :dw]
                else: tracker.clear(color)
            else:
                tracker.clear(color)
                confirmed.pop(color, None)

        # UI Rendering
        bgr_map = {"Red": (0,0,255), "Green": (0,255,0), "Blue": (255,0,0)}
        for color, bgr in bgr_map.items():
            if color in current_frame_results:
                res = current_frame_results[color]
                x, y, w, h = res["box"]
                thick = 3 if color in confirmed else 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), bgr, thick)
                label = f"{color[0]}:{res['val']}" + (" OK" if color in confirmed else "...")
                cv2.putText(frame, label, (x, y-10), 2, 0.6, bgr, 2)

        # 4. Formulas & Results
        if len(confirmed) == 3:
            idx = calculate_index(confirmed["Red"], confirmed["Green"], confirmed["Blue"])
            final = calculate_final(idx, s["top_number"])
            cv2.rectangle(frame, (10, 10), (320, 110), (0,0,0), -1)
            cv2.putText(frame, f"INDEX: {idx}", (20, 45), 2, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"FINAL: {final}", (20, 90), 2, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Stabilizing...", (10, 30), 2, 0.6, (0,0,255), 1)

        cv2.imshow("Dice Detector Pro", frame)
        
        # 5. Keyboard Shortcuts
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('d'): debug_mode = not debug_mode
        if key == ord('r'): 
            confirmed.clear()
            for k in tracker.history: tracker.history[k].clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dice Detector Pro")
    parser.add_argument("--camera", type=str, default="0", help="Camera index or URL (default: 0)")
    args = parser.parse_args()
    
    # Try to convert to int if it's a digit
    camera_param = args.camera
    if camera_param.isdigit():
        camera_param = int(camera_param)
        
    main(camera_param)