"""
Pi Camera Airy Disk Tracker — Flask web UI backend  (with PSF analysis)
=======================================================================
Serves:
  GET  /           → tracking UI (templates/index.html)
  WS   /ws/frame   → binary JPEG frames pushed at TARGET_FPS over WebSocket
  GET  /stream     → legacy MJPEG stream (fallback / VLC / ffmpeg)
  GET  /frame      → single JPEG snapshot (fallback for non-WS clients)
  GET  /state      → JSON tracking + PSF state (~12 Hz poll)
  POST /control    → UI controls (reset dots, threshold)

Open  http://<pi-ip>:8080  in any browser on the same network.

PSF metrics computed each frame (pure numpy, no astropy):
  peak, flux, background, snr,
  sigma_x/y, fwhm_x/y, fwhm_mean, ellipticity, angle_deg

Dependencies: flask, flask-sock, picamera2, cv2, numpy
"""

from picamera2 import Picamera2
import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from time import sleep, perf_counter
import threading
import json

from flask import Flask, render_template, jsonify, Response, request
from flask_sock import Sock

# ─────────────────────────── Configuration ───────────────────────────────────

FRAME_WIDTH      = 640
FRAME_HEIGHT     = 480
TARGET_FPS       = 60.0
WEB_PORT         = 8080
JPEG_QUALITY     = 80

BLOB_MIN_AREA    = 200
BLOB_MAX_AREA    = 40_000
BLOB_MIN_CIRC    = 0.40
GAUSSIAN_KERNEL  = 11
THRESH_VAL       = 130

INTERSECTION_DIST = 50
APERTURE_RADIUS   = 24        # px — centroid + PSF signal aperture
BG_INNER_RADIUS   = 28        # px — background annulus inner edge
BG_OUTER_RADIUS   = 36        # px — background annulus outer edge
VELOCITY_WINDOW   = 5
TRAIL_LENGTH      = 40

REFERENCE_POINT: Tuple[float, float] = (FRAME_WIDTH / 2, FRAME_HEIGHT / 2)

# BGR colours
COL_A    = (255, 220,   0)    # gold
COL_B    = (  0, 255, 200)    # teal
COL_REF  = (  0, 220, 255)    # cyan
COL_WARN = (  0,  90, 255)    # orange-red
COL_PSF  = (180,  80, 255)    # purple — PSF ellipse

# ─────────────────────────── Flask app ───────────────────────────────────────

app  = Flask(__name__)
sock = Sock(app)

# ─────────────────────────── Shared state ────────────────────────────────────

_lock           = threading.Lock()
_jpeg_frame     = b""
_tracking_state: Dict[str, Any] = {}
_thresh         = THRESH_VAL
_frame_idx      = 0
_fps_actual     = 0.0
_intersecting   = False
_needs_reset    = False

# ─────────────────────────── DotState ────────────────────────────────────────

@dataclass
class DotState:
    label: str
    pos: np.ndarray
    vel: np.ndarray    = field(default_factory=lambda: np.zeros(2))
    pos_history: deque = field(default_factory=lambda: deque(maxlen=VELOCITY_WINDOW + 1))
    trail:       deque = field(default_factory=lambda: deque(maxlen=TRAIL_LENGTH))
    predicted:   bool  = False
    lost_frames: int   = 0
    psf: Dict[str, float] = field(default_factory=dict)

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


# ─────────────────────────── PSF measurement ─────────────────────────────────

def measure_psf(gray: np.ndarray,
                x: float, y: float,
                sig_r: float = APERTURE_RADIUS,
                bg_inner: float = BG_INNER_RADIUS,
                bg_outer: float = BG_OUTER_RADIUS) -> Dict[str, float]:
    r  = int(np.ceil(bg_outer))
    x0 = max(0, int(x) - r);  x1 = min(gray.shape[1], int(x) + r + 1)
    y0 = max(0, int(y) - r);  y1 = min(gray.shape[0], int(y) + r + 1)
    roi = gray[y0:y1, x0:x1].astype(np.float64)
    if roi.size == 0:
        return {}

    rows, cols = np.mgrid[y0:y1, x0:x1]
    dx = cols - x
    dy = rows - y
    r2 = dx ** 2 + dy ** 2

    ap_mask  = r2 <= sig_r   ** 2
    bg_mask  = (r2 >= bg_inner ** 2) & (r2 <= bg_outer ** 2)

    bg_pixels = roi[bg_mask]
    if bg_pixels.size < 4:
        background = 0.0
        bg_rms     = 1.0
    else:
        background = float(np.median(bg_pixels))
        bg_rms     = float(np.std(bg_pixels)) if np.std(bg_pixels) > 0 else 1.0

    ap_pixels = roi[ap_mask]
    peak      = float(ap_pixels.max()) if ap_pixels.size else 0.0
    flux      = float((ap_pixels - background).sum())
    snr       = (peak - background) / bg_rms if bg_rms > 0 else 0.0

    w     = np.maximum(roi - background, 0) * ap_mask
    total = w.sum()
    if total == 0:
        return {
            "peak": peak, "flux": flux, "background": background, "snr": snr,
            "sigma_x": 0.0, "sigma_y": 0.0,
            "fwhm_x": 0.0, "fwhm_y": 0.0, "fwhm_mean": 0.0,
            "ellipticity": 0.0, "angle_deg": 0.0,
        }

    mu_x  = (w * dx).sum() / total
    mu_y  = (w * dy).sum() / total
    dxc   = dx - mu_x
    dyc   = dy - mu_y

    m20 = (w * dxc ** 2).sum() / total
    m02 = (w * dyc ** 2).sum() / total
    m11 = (w * dxc * dyc).sum() / total

    sigma_x   = float(np.sqrt(max(m20, 0)))
    sigma_y   = float(np.sqrt(max(m02, 0)))
    fwhm_x    = 2.3548 * sigma_x
    fwhm_y    = 2.3548 * sigma_y
    fwhm_mean = float(np.sqrt(fwhm_x * fwhm_y)) if fwhm_x * fwhm_y > 0 else 0.0

    trace  = m20 + m02
    det    = m20 * m02 - m11 ** 2
    disc   = max((trace / 2) ** 2 - det, 0)
    lam1   = trace / 2 + np.sqrt(disc)
    lam2   = trace / 2 - np.sqrt(disc)
    sigma_major = float(np.sqrt(max(lam1, 0)))
    sigma_minor = float(np.sqrt(max(lam2, 0)))
    ellipticity = (sigma_major - sigma_minor) / sigma_major if sigma_major > 0 else 0.0
    if abs(m11) < 1e-9 and m20 >= m02:
        angle_deg = 0.0
    elif abs(m11) < 1e-9:
        angle_deg = 90.0
    else:
        angle_deg = float(np.degrees(0.5 * np.arctan2(2 * m11, m20 - m02)))

    return {
        "peak":        round(peak,         1),
        "flux":        round(flux,         1),
        "background":  round(background,   2),
        "snr":         round(snr,          2),
        "sigma_x":     round(sigma_x,      2),
        "sigma_y":     round(sigma_y,      2),
        "fwhm_x":      round(fwhm_x,       2),
        "fwhm_y":      round(fwhm_y,       2),
        "fwhm_mean":   round(fwhm_mean,    2),
        "ellipticity": round(ellipticity,  3),
        "angle_deg":   round(angle_deg,    1),
    }


# ─────────────────────────── Centroid & detection ────────────────────────────

def circular_aperture_centroid(gray, x, y, radius=APERTURE_RADIUS):
    r  = int(np.ceil(radius))
    x0 = max(0, int(x) - r);  x1 = min(gray.shape[1], int(x) + r + 1)
    y0 = max(0, int(y) - r);  y1 = min(gray.shape[0], int(y) + r + 1)
    roi = gray[y0:y1, x0:x1].astype(np.float64)
    if roi.size == 0:
        return None
    rows, cols = np.mgrid[y0:y1, x0:x1]
    mask    = ((cols - x) ** 2 + (rows - y) ** 2) <= radius ** 2
    weights = roi * mask
    total   = weights.sum()
    if total == 0:
        return None
    cx = (weights * cols).sum() / total
    cy = (weights * rows).sum() / total
    return np.array([cx, cy]) if (np.isfinite(cx) and np.isfinite(cy)) else None


def blob_peak_brightness(gray, pos, radius=APERTURE_RADIUS):
    r  = int(np.ceil(radius))
    x0 = max(0, int(pos[0]) - r);  x1 = min(gray.shape[1], int(pos[0]) + r + 1)
    y0 = max(0, int(pos[1]) - r);  y1 = min(gray.shape[0], int(pos[1]) + r + 1)
    roi = gray[y0:y1, x0:x1]
    return float(roi.max()) if roi.size else 0.0


def detect_blobs(gray, thresh):
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
    kps = detector.detect(cv2.bitwise_not(blurred))
    return [np.array(kp.pt) for kp in kps]


def find_bright_peaks(gray, n=2):
    """Find n brightest isolated spots using brightness threshold + connected components.

    More robust than SimpleBlobDetector for reseeding: adaptive threshold descends
    until at least n candidates are found, then returns the brightest n sorted left→right.
    """
    blurred = cv2.GaussianBlur(gray, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
    max_val = int(blurred.max())
    if max_val < 20:
        return []

    candidates = []
    for pct in (0.75, 0.60, 0.45, 0.30, 0.20):
        thresh_val = max(20, int(max_val * pct))
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
        num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        candidates = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if BLOB_MIN_AREA <= area <= BLOB_MAX_AREA:
                cx, cy = centroids[i]
                brightness = blob_peak_brightness(blurred, np.array([cx, cy]))
                refined = circular_aperture_centroid(gray, cx, cy)
                pos = refined if refined is not None else np.array([cx, cy])
                candidates.append((brightness, pos))
        if len(candidates) >= n:
            break

    candidates.sort(key=lambda t: t[0], reverse=True)
    return [pos for _, pos in candidates[:n]]


def nearest_blob(dot, blobs, used, max_dist):
    best_i, best_d, best_b = None, np.inf, None
    for i, b in enumerate(blobs):
        if i in used:
            continue
        d = np.linalg.norm(b - dot.pos)
        if d < best_d:
            best_i, best_d, best_b = i, d, b
    return (best_i, best_b) if (best_i is not None and best_d < max_dist) else None


def auto_init_dots(gray, thresh):
    """Seed Dot A/B from the two brightest spots.

    Uses brightness-based peak finding (adaptive threshold + connected components)
    as the primary method, falling back to SimpleBlobDetector if needed.
    Dots are sorted left→right (Dot A = left, Dot B = right).
    """
    peaks = find_bright_peaks(gray, n=2)
    if len(peaks) >= 2:
        peaks.sort(key=lambda p: float(p[0]))
        return peaks[0], peaks[1]
    if len(peaks) == 1:
        return peaks[0], None

    # Fallback: SimpleBlobDetector
    blobs = detect_blobs(gray, thresh)
    if not blobs:
        return None, None
    candidates = []
    for b in blobs:
        c = circular_aperture_centroid(gray, b[0], b[1])
        pos = c if c is not None else b
        candidates.append((blob_peak_brightness(gray, pos), pos))
    candidates.sort(key=lambda t: t[0], reverse=True)
    top = [pos for _, pos in candidates[:2]]
    if len(top) == 2:
        top.sort(key=lambda p: float(p[0]))
        return top[0], top[1]
    return (top[0], None) if top else (None, None)


# ─────────────────────────── Frame drawing ───────────────────────────────────

def draw_trail(img, trail, color):
    pts = list(trail)
    n   = len(pts)
    for i in range(1, n):
        alpha = i / n
        c = tuple(int(ch * alpha) for ch in color)
        cv2.line(img, pts[i - 1], pts[i], c, 1, cv2.LINE_AA)


def draw_psf_ellipse(img, dot, col):
    psf = dot.psf
    cx, cy = int(dot.pos[0]), int(dot.pos[1])
    sx = psf.get("sigma_x", 0)
    sy = psf.get("sigma_y", 0)
    angle = psf.get("angle_deg", 0)
    if sx < 0.5 or sy < 0.5:
        return
    axes = (max(2, int(round(2 * sx))), max(2, int(round(2 * sy))))
    cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, COL_PSF, 1, cv2.LINE_AA)


def draw_overlay(gray_frame: np.ndarray,
                 dots: List[DotState],
                 intersecting: bool) -> np.ndarray:
    out = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    ref = np.array(REFERENCE_POINT)
    rx, ry = int(ref[0]), int(ref[1])

    cv2.drawMarker(out, (rx, ry), COL_REF,
                   markerType=cv2.MARKER_CROSS, markerSize=22,
                   thickness=2, line_type=cv2.LINE_AA)

    dot_colors = [COL_A, COL_B]
    for dot, col in zip(dots, dot_colors):
        cx, cy = int(dot.pos[0]), int(dot.pos[1])
        c = col if not dot.predicted else COL_WARN

        draw_trail(out, dot.trail, c)
        cv2.circle(out, (cx, cy), APERTURE_RADIUS, c, 1, cv2.LINE_AA)
        draw_psf_ellipse(out, dot, col)
        cv2.drawMarker(out, (cx, cy), c,
                       markerType=cv2.MARKER_CROSS, markerSize=16,
                       thickness=2, line_type=cv2.LINE_AA)
        cv2.line(out, (cx, cy), (rx, ry), c, 1, cv2.LINE_AA)

    if intersecting:
        cv2.rectangle(out, (0, 0), (FRAME_WIDTH, 3), COL_WARN, -1)

    return out


# ─────────────────────────── Capture thread ──────────────────────────────────

def capture_loop():
    global _jpeg_frame, _tracking_state, _thresh, _frame_idx
    global _fps_actual, _intersecting, _needs_reset

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    sleep(0.5)
    picam2.set_controls({"FrameRate": TARGET_FPS})

    dot_a = dot_b = None
    print("Waiting for initial blobs…")
    for _ in range(60):
        rgb  = picam2.capture_array()
        gray = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        pa, pb = auto_init_dots(gray, _thresh)
        if pa is not None and pb is not None:
            dot_a = DotState("Dot A", pa)
            dot_b = DotState("Dot B", pb)
            for dot, p in [(dot_a, pa), (dot_b, pb)]:
                dot.pos_history.append(p.copy())
                dot.trail.append(tuple(p.astype(int)))
            print(f"  Dot A → ({pa[0]:.1f}, {pa[1]:.1f})")
            print(f"  Dot B → ({pb[0]:.1f}, {pb[1]:.1f})")
            break
        sleep(0.05)
    else:
        cx, cy = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
        dot_a = DotState("Dot A", np.array([cx - 60.0, cy], dtype=float))
        dot_b = DotState("Dot B", np.array([cx + 60.0, cy], dtype=float))
        print("[WARN] No blobs at startup — dots placed at centre.")

    dots   = [dot_a, dot_b]
    t_last = perf_counter()
    print(f"Tracker running — open http://localhost:{WEB_PORT} in a browser.")

    while True:
        rgb  = picam2.capture_array()
        gray = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)

        t_now  = perf_counter()
        fps    = 1.0 / max(t_now - t_last, 1e-6)
        t_last = t_now

        if _needs_reset:
            pa, pb = auto_init_dots(gray, _thresh)
            if pa is not None:
                dots[0] = DotState("Dot A", pa)
                dots[0].pos_history.append(pa.copy())
                dots[0].trail.append(tuple(pa.astype(int)))
            if pb is not None:
                dots[1] = DotState("Dot B", pb)
                dots[1].pos_history.append(pb.copy())
                dots[1].trail.append(tuple(pb.astype(int)))
            _needs_reset = False

        raw     = detect_blobs(gray, _thresh)
        refined = []
        for b in raw:
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

        for dot in dots:
            dot.psf = measure_psf(gray, dot.pos[0], dot.pos[1])

        frame_out = draw_overlay(gray, dots, intersecting)
        _, jpeg = cv2.imencode(".jpg", frame_out,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        ref = np.array(REFERENCE_POINT)
        state: Dict[str, Any] = {
            "frame":         _frame_idx,
            "fps":           round(fps, 1),
            "thresh":        _thresh,
            "intersecting":  intersecting,
            "separation_px": round(sep, 1),
            "dots": [],
        }
        for dot in dots:
            ev = dot.pos - ref
            state["dots"].append({
                "label":       dot.label,
                "x":           round(float(dot.pos[0]), 2),
                "y":           round(float(dot.pos[1]), 2),
                "err_x":       round(float(ev[0]), 2),
                "err_y":       round(float(ev[1]), 2),
                "err_mag":     round(float(np.linalg.norm(ev)), 2),
                "vel_x":       round(float(dot.vel[0]), 2),
                "vel_y":       round(float(dot.vel[1]), 2),
                "predicted":   dot.predicted,
                "lost_frames": dot.lost_frames,
                "psf":         dot.psf,
            })

        with _lock:
            _jpeg_frame     = jpeg.tobytes()
            _tracking_state = state
            _frame_idx     += 1
            _fps_actual     = fps
            _intersecting   = intersecting


# ─────────────────────────── Flask routes ────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/frame")
def frame():
    """Single annotated JPEG frame — polled by browser JS at ~30 fps."""
    with _lock:
        data = _jpeg_frame
    if not data:
        return Response("Frame not ready", status=503)
    return Response(
        data,
        mimetype="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@sock.route("/ws/frame")
def ws_frame(ws):
    """Push annotated JPEG frames as binary WebSocket messages at TARGET_FPS."""
    while True:
        with _lock:
            data = _jpeg_frame
        if data:
            try:
                ws.send(data)
            except Exception:
                break
        sleep(1 / TARGET_FPS)


@app.route("/stream")
def stream():
    """Legacy MJPEG stream — kept for backward compatibility."""
    def generate():
        while True:
            with _lock:
                data = _jpeg_frame
            if data:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(data)).encode() + b"\r\n"
                    b"\r\n" + data + b"\r\n"
                )
            sleep(1 / TARGET_FPS)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )


@app.route("/state")
def state():
    with _lock:
        data = dict(_tracking_state)
    return jsonify(data)


@app.route("/control", methods=["POST"])
def control():
    global _needs_reset, _thresh
    body = request.get_json(silent=True) or {}
    if body.get("action") == "reset":
        _needs_reset = True
    if "thresh" in body:
        _thresh = max(20, min(250, int(body["thresh"])))
    return jsonify({"ok": True})


# ─────────────────────────── Entry point ─────────────────────────────────────

if __name__ == "__main__":
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    print(f"HTTP server on port {WEB_PORT}")
    app.run(host="0.0.0.0", port=WEB_PORT, threaded=True)
