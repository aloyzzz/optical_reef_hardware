"""
Pi Camera Airy Disk Tracker  (no astropy / photutils dependency)
=================================================================
Live Airy-disk dot tracker on a Raspberry Pi camera feed.
Sub-pixel centroiding is done with a pure-numpy intensity-weighted
centroid inside a circular aperture mask — same maths as photutils
ApertureStats, zero extra dependencies beyond cv2 + numpy.
 
Controls:
    Q / ESC  → quit
    R        → reset dot positions to current auto-detected blobs
    S        → toggle static-rejection ON / OFF
    +/-      → raise / lower detection threshold on the fly
 
Usage:
    python pi_disk_tracker.py
"""
 
from picamera2 import Picamera2
import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from time import sleep, perf_counter
 
# ─────────────────────────── Configuration ───────────────────────────────────
 
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
TARGET_FPS   = 60.0
 
# Known static blobs to ignore (pixel coords for your lens/scene).
# Populate after a first run; leave empty to disable.
STATIC_POSITIONS: List[np.ndarray] = [
    # np.array([320.0, 240.0]),
]
STATIC_RADIUS = 40      # px — blobs within this of a static position are skipped
 
# Blob detection
BLOB_MIN_AREA   = 200
BLOB_MAX_AREA   = 40_000
BLOB_MIN_CIRC   = 0.40
GAUSSIAN_KERNEL = 11
THRESH_VAL      = 130   # adjustable live with +/- keys
 
# Intersection / occlusion
INTERSECTION_DIST = 50  # px — below this, coast both dots on inertia
 
# Aperture radius for centroid refinement (also drawn on screen)
APERTURE_RADIUS = 24
 
# Velocity averaging & trail
VELOCITY_WINDOW = 5
TRAIL_LENGTH    = 40
 
# Reference crosshair (frame centre by default)
REFERENCE_POINT: Tuple[float, float] = (FRAME_WIDTH / 2, FRAME_HEIGHT / 2)
 
# Colours (BGR)
COL_A     = (255, 220,   0)   # gold
COL_B     = (  0, 255, 200)   # teal
COL_REF   = (  0, 220, 255)   # cyan
COL_WARN  = (  0,  90, 255)   # orange
COL_WHITE = (255, 255, 255)
COL_BLACK = (  0,   0,   0)
COL_GRAY  = (160, 160, 160)
COL_GREEN = ( 50, 255,  80)
 
# ─────────────────────────── DotState ────────────────────────────────────────
 
@dataclass
class DotState:
    label: str
    color: Tuple[int, int, int]
    pos: np.ndarray
    vel: np.ndarray        = field(default_factory=lambda: np.zeros(2))
    pos_history: deque     = field(default_factory=lambda: deque(maxlen=VELOCITY_WINDOW + 1))
    trail:       deque     = field(default_factory=lambda: deque(maxlen=TRAIL_LENGTH))
    predicted:   bool      = False
    lost_frames: int       = 0
 
    def update_position(self, new_pos: np.ndarray) -> None:
        self.pos_history.append(new_pos.copy())
        if len(self.pos_history) >= 2:
            disps = [
                np.array(self.pos_history[i + 1]) - np.array(self.pos_history[i])
                for i in range(len(self.pos_history) - 1)
            ]
            self.vel = np.mean(disps, axis=0)
        self.pos         = new_pos.copy()
        self.trail.append(tuple(new_pos.astype(int)))
        self.predicted   = False
        self.lost_frames = 0
 
    def predict_next(self) -> np.ndarray:
        predicted    = self.pos + self.vel
        predicted[0] = np.clip(predicted[0], 0, FRAME_WIDTH  - 1)
        predicted[1] = np.clip(predicted[1], 0, FRAME_HEIGHT - 1)
        self.pos     = predicted.copy()
        self.trail.append(tuple(predicted.astype(int)))
        self.predicted    = True
        self.lost_frames += 1
        return predicted
 
 
# ─────────────────────────── Centroid (pure numpy) ───────────────────────────
 
def circular_aperture_centroid(gray: np.ndarray,
                               x: float, y: float,
                               radius: float = APERTURE_RADIUS
                               ) -> Optional[np.ndarray]:
    """
    Intensity-weighted centroid inside a circular aperture.
 
    Equivalent to photutils ApertureStats(data, CircularAperture).centroid
    but implemented with pure numpy — no astropy dependency.
 
    Steps
    -----
    1. Extract a bounding-box ROI around (x, y) with radius r.
    2. Build a boolean circular mask (pixels whose centre falls inside r).
    3. Compute the intensity-weighted mean of the (col, row) pixel coordinates
       — this is the standard first-moment / centre-of-mass centroid.
    4. Return absolute (x, y) in full-frame coordinates.
    """
    r  = int(np.ceil(radius))
    # ROI bounds (clamped to frame)
    x0 = max(0, int(x) - r);  x1 = min(gray.shape[1], int(x) + r + 1)
    y0 = max(0, int(y) - r);  y1 = min(gray.shape[0], int(y) + r + 1)
    roi = gray[y0:y1, x0:x1].astype(np.float64)
 
    if roi.size == 0:
        return None
 
    # Pixel-coordinate grids relative to aperture centre
    rows, cols = np.mgrid[y0:y1, x0:x1]
    dy = rows - y
    dx = cols - x
 
    # Circular mask
    mask = (dx ** 2 + dy ** 2) <= radius ** 2
    weights = roi * mask
 
    total = weights.sum()
    if total == 0:
        return None
 
    cx = (weights * cols).sum() / total
    cy = (weights * rows).sum() / total
 
    if np.isfinite(cx) and np.isfinite(cy):
        return np.array([cx, cy])
    return None
 
 
# ─────────────────────────── Blob detection ──────────────────────────────────
 
def detect_blobs(gray: np.ndarray, thresh: int) -> List[np.ndarray]:
    """Detect significant circular blobs via SimpleBlobDetector."""
    blurred = cv2.GaussianBlur(gray, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea        = True
    params.minArea             = BLOB_MIN_AREA
    params.maxArea             = BLOB_MAX_AREA
    params.filterByCircularity = True
    params.minCircularity      = BLOB_MIN_CIRC
    params.filterByInertia     = False
    params.filterByConvexity   = False
    params.minThreshold        = max(20, thresh - 60)
    params.maxThreshold        = 255
    detector = cv2.SimpleBlobDetector_create(params)
    inv = cv2.bitwise_not(blurred)
    kps = detector.detect(inv)
    return [np.array(kp.pt) for kp in kps]
 
 
def reject_static(blobs: List[np.ndarray], enabled: bool = True) -> List[np.ndarray]:
    if not enabled or not STATIC_POSITIONS:
        return blobs
    return [
        b for b in blobs
        if all(np.linalg.norm(b - s) > STATIC_RADIUS for s in STATIC_POSITIONS)
    ]
 
 
def nearest_blob(dot: DotState,
                 blobs: List[np.ndarray],
                 used: set,
                 max_dist: float) -> Optional[Tuple[int, np.ndarray]]:
    best_i, best_d, best_b = None, np.inf, None
    for i, b in enumerate(blobs):
        if i in used:
            continue
        d = np.linalg.norm(b - dot.pos)
        if d < best_d:
            best_i, best_d, best_b = i, d, b
    if best_i is not None and best_d < max_dist:
        return best_i, best_b
    return None
 
 
def auto_init_dots(gray: np.ndarray,
                   thresh: int,
                   static_on: bool = True
                   ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Seed Dot A/B from the first two detected blobs (sorted left→right)."""
    blobs   = reject_static(detect_blobs(gray, thresh), static_on)
    refined = []
    for b in blobs:
        c = circular_aperture_centroid(gray, b[0], b[1])
        refined.append(c if c is not None else b)
    refined = refined[:2]
    if len(refined) == 2:
        refined.sort(key=lambda p: p[0])   # A = left, B = right
        return refined[0], refined[1]
    if len(refined) == 1:
        return refined[0], None
    return None, None
 
 
# ─────────────────────────── Drawing helpers ─────────────────────────────────
 
def draw_trail(img: np.ndarray, trail: deque, color: Tuple) -> None:
    pts = list(trail)
    n   = len(pts)
    for i in range(1, n):
        alpha = i / n
        c = tuple(int(ch * alpha) for ch in color)
        cv2.line(img, pts[i - 1], pts[i], c, 1, cv2.LINE_AA)
 
 
def shadow_text(img, text, pos, scale=0.48, color=COL_WHITE, thickness=1):
    ox, oy = pos
    cv2.putText(img, text, (ox + 1, oy + 1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, COL_BLACK, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)
 
 
def draw_hud(img: np.ndarray,
             dots: List[DotState],
             frame_idx: int,
             intersecting: bool,
             fps_actual: float,
             thresh: int,
             static_enabled: bool) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    ref  = np.array(REFERENCE_POINT)
    rx, ry = int(ref[0]), int(ref[1])
 
    # ── Reference / target crosshair ─────────────────────────────────────────
    cv2.drawMarker(out, (rx, ry), COL_REF,
                   markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2,
                   line_type=cv2.LINE_AA)
    shadow_text(out, "TARGET", (rx + 14, ry - 10), scale=0.42, color=COL_REF)
 
    # ── Per-dot overlays ──────────────────────────────────────────────────────
    for dot in dots:
        cx, cy  = int(dot.pos[0]), int(dot.pos[1])
        col     = dot.color if not dot.predicted else COL_WARN
        err_vec = dot.pos - ref
        err_mag = float(np.linalg.norm(err_vec))
        vx, vy  = dot.vel
 
        draw_trail(out, dot.trail, col)
 
        cv2.circle(out, (cx, cy), APERTURE_RADIUS, col, 1, cv2.LINE_AA)
 
        cv2.drawMarker(out, (cx, cy), col,
                       markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2,
                       line_type=cv2.LINE_AA)
 
        cv2.line(out, (cx, cy), (rx, ry), col, 1, cv2.LINE_AA)
 
        lx = min(cx + 36, w - 210)
        ly = max(cy - 28, 18)
        tag = dot.label + ("  [PREDICTED]" if dot.predicted else "")
        shadow_text(out, tag,                                      (lx, ly),      scale=0.50, color=col)
        shadow_text(out, f"pos  x={cx}  y={cy}",                  (lx, ly + 18), scale=0.40, color=COL_WHITE)
        shadow_text(out, f"err  dx={err_vec[0]:+.1f}  dy={err_vec[1]:+.1f}",
                                                                   (lx, ly + 34), scale=0.40, color=COL_WHITE)
        shadow_text(out, f"|err| {err_mag:.1f} px",               (lx, ly + 50), scale=0.40, color=COL_WHITE)
        shadow_text(out, f"vel  vx={vx:+.1f}  vy={vy:+.1f} px/fr",
                                                                   (lx, ly + 66), scale=0.38, color=COL_GRAY)
        if dot.lost_frames > 0:
            shadow_text(out, f"lost {dot.lost_frames} fr",        (lx, ly + 82), scale=0.38, color=COL_WARN)
 
    # ── Intersection banner ───────────────────────────────────────────────────
    if intersecting:
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)
        shadow_text(out, "INTERSECTION — coasting on inertia",
                    (12, 24), scale=0.58, color=COL_WARN, thickness=1)
 
    # ── Status bar ────────────────────────────────────────────────────────────
    bar_y = h - 10
    shadow_text(out, f"Frame {frame_idx}",                         (10,  bar_y), scale=0.40, color=COL_GRAY)
    shadow_text(out, f"FPS {fps_actual:.1f}",                      (100, bar_y), scale=0.40, color=COL_GREEN)
    shadow_text(out, f"THRESH {thresh}",                           (185, bar_y), scale=0.40, color=COL_GRAY)
    static_lbl = "STATIC-REJ ON" if static_enabled else "STATIC-REJ OFF"
    shadow_text(out, static_lbl,                                   (270, bar_y), scale=0.40,
                color=COL_GREEN if static_enabled else COL_WARN)
    shadow_text(out, "centroid:numpy-aperture",                    (420, bar_y), scale=0.38, color=COL_GRAY)
 
    # ── Key-bindings legend ───────────────────────────────────────────────────
    for i, line in enumerate(["Q/ESC quit", "R reset", "S static-rej", "+/- thresh"]):
        shadow_text(out, line, (w - 145, 18 + i * 16), scale=0.36, color=COL_GRAY)
 
    return out
 
 
# ─────────────────────────── Main ────────────────────────────────────────────
 
def main():
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    sleep(0.5)
    picam2.set_controls({"FrameRate": TARGET_FPS})
 
    thresh         = THRESH_VAL
    static_enabled = True
    frame_idx      = 0
    fps_actual     = 0.0
    t_last         = perf_counter()
 
    # ── Auto-seed from first two blobs ────────────────────────────────────────
    print("Waiting for initial blobs…")
    dot_a = dot_b = None
 
    for _ in range(30):
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gray      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        pa, pb    = auto_init_dots(gray, thresh, static_enabled)
        if pa is not None and pb is not None:
            dot_a = DotState("Dot A", COL_A, pa)
            dot_b = DotState("Dot B", COL_B, pb)
            for dot, p in [(dot_a, pa), (dot_b, pb)]:
                dot.pos_history.append(p.copy())
                dot.trail.append(tuple(p.astype(int)))
            print(f"  Dot A → ({pa[0]:.1f}, {pa[1]:.1f})")
            print(f"  Dot B → ({pb[0]:.1f}, {pb[1]:.1f})")
            break
        sleep(0.05)
    else:
        cx, cy = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
        dot_a = DotState("Dot A", COL_A, np.array([cx - 60.0, cy], dtype=float))
        dot_b = DotState("Dot B", COL_B, np.array([cx + 60.0, cy], dtype=float))
        print("[WARN] No blobs found at startup — dots placed at frame centre.")
 
    dots = [dot_a, dot_b]
    print("Running — press Q or ESC to quit.")
    cv2.namedWindow("Airy Disk Tracker", cv2.WINDOW_NORMAL)
 
    while True:
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gray      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
 
        t_now      = perf_counter()
        fps_actual = 1.0 / max(t_now - t_last, 1e-6)
        t_last     = t_now
 
        # Detect, filter static, refine centroids
        raw_blobs = detect_blobs(gray, thresh)
        blobs     = reject_static(raw_blobs, static_enabled)
        refined   = []
        for b in blobs:
            c = circular_aperture_centroid(gray, b[0], b[1])
            refined.append(c if c is not None else b)
 
        sep          = float(np.linalg.norm(dots[0].pos - dots[1].pos))
        intersecting = sep < INTERSECTION_DIST
 
        if intersecting:
            for dot in dots:
                dot.predict_next()
        else:
            used = set()
            for dot in dots:
                max_d = APERTURE_RADIUS * (6 if dot.predicted else 3)
                match = nearest_blob(dot, refined, used, max_d)
                if match is not None:
                    idx, blob_pos = match
                    used.add(idx)
                    dot.update_position(blob_pos)
                else:
                    dot.predict_next()
 
        annotated = draw_hud(frame_bgr, dots, frame_idx,
                             intersecting, fps_actual, thresh, static_enabled)
        cv2.imshow("Airy Disk Tracker", annotated)
        frame_idx += 1
 
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            pa, pb = auto_init_dots(gray, thresh, static_enabled)
            if pa is not None:
                dots[0] = DotState("Dot A", COL_A, pa)
                dots[0].pos_history.append(pa.copy())
                dots[0].trail.append(tuple(pa.astype(int)))
            if pb is not None:
                dots[1] = DotState("Dot B", COL_B, pb)
                dots[1].pos_history.append(pb.copy())
                dots[1].trail.append(tuple(pb.astype(int)))
            print(f"[R] Reset → A=({dots[0].pos[0]:.1f},{dots[0].pos[1]:.1f})  "
                  f"B=({dots[1].pos[0]:.1f},{dots[1].pos[1]:.1f})")
        elif key == ord('s'):
            static_enabled = not static_enabled
            print(f"[S] Static rejection: {'ON' if static_enabled else 'OFF'}")
        elif key in (ord('+'), ord('=')):
            thresh = min(thresh + 5, 250)
            print(f"[+] Threshold → {thresh}")
        elif key == ord('-'):
            thresh = max(thresh - 5, 20)
            print(f"[-] Threshold → {thresh}")
 
    cv2.destroyAllWindows()
    picam2.stop()
    print("Done.")
 
 
if __name__ == "__main__":
    main()