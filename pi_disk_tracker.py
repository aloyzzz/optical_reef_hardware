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

Dependencies: flask, flask-sock, picamera2, cv2, numpy, scipy
"""

from picamera2 import Picamera2
import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from time import sleep, perf_counter
import threading

from flask import Flask, render_template, jsonify, Response, request
from flask_sock import Sock
from scipy.optimize import linear_sum_assignment

# ─────────────────────────── Configuration ───────────────────────────────────

FRAME_WIDTH      = 640
FRAME_HEIGHT     = 480
TARGET_FPS       = 60.0
WEB_PORT         = 8080
JPEG_QUALITY     = 80

BLOB_MIN_AREA    = 30
BLOB_MAX_AREA    = 8_000
BLOB_MIN_CIRC    = 0.35
GAUSSIAN_KERNEL  = 5
THRESH_VAL       = 25

# Top-hat structuring element — must be larger than the dots, smaller than background
# variation scale. 41 px works for ~10-15 px diameter Airy disks on a slowly-varying
# background. Increase if background gradients are very broad.
TOPHAT_KERNEL    = 41

INTERSECTION_DIST          = 50
INTERSECTION_SEARCH_RADIUS = 60   # px — ROI radius when tracking merged blob
RING_EDGES = [0, 6, 12, 20, 30]  # px — concentric ring boundaries for intersection analysis
APERTURE_RADIUS   = 15        # px — centroid + PSF signal aperture (tuned for ~15px Airy disks)
BG_INNER_RADIUS   = 17        # px — background annulus inner edge
BG_OUTER_RADIUS   = 25        # px — background annulus outer edge
TRAIL_LENGTH      = 40

# ── Runtime-calibrated physics parameters (overwritten by calibrate_tracking_params) ──
_ap_r            = float(APERTURE_RADIUS)
_bg_inner_r      = float(BG_INNER_RADIUS)
_bg_outer_r      = float(BG_OUTER_RADIUS)
_intersect_dist  = float(INTERSECTION_DIST)
_intersect_sr    = float(INTERSECTION_SEARCH_RADIUS)
_ring_edges_d    = list(RING_EDGES)
_max_assign_cost = float(APERTURE_RADIUS * 8)

REFERENCE_POINT: Tuple[float, float] = (FRAME_WIDTH / 2, FRAME_HEIGHT / 2)

# ── Kalman filter constants (constant-velocity model, state = [x, y, vx, vy]) ─
KF_F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=float)
KF_H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
KF_Q = np.diag([0.5, 0.5, 2.0, 2.0])   # process noise: smooth motion at 60 fps
KF_R = np.diag([9.0, 9.0])              # measurement noise: ±3 px (low-contrast centroid)
KF_INIT_COV = 100.0                     # initial state uncertainty

# ── PSF-based identity matching ────────────────────────────────────────────────
PSF_EMA_ALPHA       = 0.10              # PSF reference adaptation rate
PSF_WEIGHT          = 60.0             # px-equivalent cost per unit PSF dissimilarity
MAX_ASSIGNMENT_COST = APERTURE_RADIUS * 8   # 192 px — reject worse assignments

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

# ─────────────────────────── Preprocessing ───────────────────────────────────

# Built once at import time; size must be larger than the dots, smaller than the
# spatial scale of background variations.
_tophat_se = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (TOPHAT_KERNEL, TOPHAT_KERNEL)
)


def preprocess_frame(gray: np.ndarray) -> np.ndarray:
    """Top-hat transform: removes slowly-varying background, leaving only small
    bright features (the dots).  Output values represent contrast above local
    background, so downstream thresholds are in contrast units, not absolute ADU.
    """
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, _tophat_se)


# ─────────────────────────── DotState ────────────────────────────────────────

@dataclass
class DotState:
    label: str
    pos:   np.ndarray
    vel:   np.ndarray         = field(default_factory=lambda: np.zeros(2))
    trail: deque              = field(default_factory=lambda: deque(maxlen=TRAIL_LENGTH))
    predicted:   bool         = False
    lost_frames: int          = 0
    psf:     Dict[str, float] = field(default_factory=dict)
    psf_ref: Dict[str, float] = field(default_factory=dict)   # EMA PSF signature

    def __post_init__(self):
        # Kalman state [x, y, vx, vy] and covariance — not dataclass fields so
        # they don't participate in __init__ signature or __repr__.
        self.kf_x = np.array([self.pos[0], self.pos[1], 0.0, 0.0])
        self.kf_P = np.eye(4) * KF_INIT_COV

    # ── Kalman predict ────────────────────────────────────────────────────────

    def kf_predict(self):
        """Propagate state one step forward; pos/vel reflect the prediction."""
        self.kf_x = KF_F @ self.kf_x
        self.kf_P = KF_F @ self.kf_P @ KF_F.T + KF_Q
        self.kf_x[0] = np.clip(self.kf_x[0], 0, FRAME_WIDTH  - 1)
        self.kf_x[1] = np.clip(self.kf_x[1], 0, FRAME_HEIGHT - 1)
        self.pos = self.kf_x[:2].copy()
        self.vel = self.kf_x[2:].copy()

    # ── Kalman update ─────────────────────────────────────────────────────────

    def kf_update(self, measurement: np.ndarray):
        """Fuse a position measurement; pos/vel reflect the posterior estimate."""
        y = measurement - KF_H @ self.kf_x
        S = KF_H @ self.kf_P @ KF_H.T + KF_R
        K = self.kf_P @ KF_H.T @ np.linalg.inv(S)
        self.kf_x = self.kf_x + K @ y
        self.kf_P = (np.eye(4) - K @ KF_H) @ self.kf_P
        self.pos = self.kf_x[:2].copy()
        self.vel = self.kf_x[2:].copy()
        self.trail.append(tuple(self.pos.astype(int)))
        self.predicted   = False
        self.lost_frames = 0

    def mark_lost(self):
        """No measurement this frame — kf_predict already ran; just bookkeep."""
        self.trail.append(tuple(self.pos.astype(int)))
        self.predicted    = True
        self.lost_frames += 1

    def freeze(self, merged_pos: Optional[np.ndarray] = None):
        """Hold at intersection blob; zero velocity to suppress Kalman drift."""
        if merged_pos is not None:
            self.kf_x[0] = merged_pos[0]
            self.kf_x[1] = merged_pos[1]
            self.kf_x[2] = 0.0
            self.kf_x[3] = 0.0
            self.pos = merged_pos.copy()
        self.kf_P        += KF_Q   # grow uncertainty so filter re-adapts on separation
        self.predicted    = True
        self.lost_frames += 1
        self.trail.append(tuple(self.pos.astype(int)))


# ─────────────────────────── PSF measurement ─────────────────────────────────

def measure_psf(gray: np.ndarray,
                x: float, y: float,
                sig_r: Optional[float] = None,
                bg_inner: Optional[float] = None,
                bg_outer: Optional[float] = None) -> Dict[str, float]:
    if sig_r   is None: sig_r   = _ap_r
    if bg_inner is None: bg_inner = _bg_inner_r
    if bg_outer is None: bg_outer = _bg_outer_r
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


# ─────────────────────────── PSF identity helpers ────────────────────────────

def psf_dissimilarity(ref: Dict[str, float], candidate: Dict[str, float]) -> float:
    """Normalised dissimilarity in [0, ∞); 0 = identical PSF signature.

    Compares peak brightness, total flux, and mean FWHM as relative differences.
    Returns 0 when either signature is missing (no penalty until reference is built).
    """
    if not ref or not candidate:
        return 0.0
    diffs = []
    for key in ('peak', 'flux', 'fwhm_mean'):
        rv = ref.get(key, 0.0)
        cv = candidate.get(key, 0.0)
        if rv > 1.0:
            diffs.append(abs(rv - cv) / rv)
    return float(np.mean(diffs)) if diffs else 0.0


def update_psf_ref(dot: DotState, new_psf: Dict[str, float]) -> None:
    """EMA-update the dot's PSF identity signature with the latest measurement."""
    if not new_psf:
        return
    if not dot.psf_ref:
        dot.psf_ref = dict(new_psf)
    else:
        for key in ('peak', 'flux', 'fwhm_mean'):
            if key in new_psf and key in dot.psf_ref:
                dot.psf_ref[key] = ((1 - PSF_EMA_ALPHA) * dot.psf_ref[key]
                                    + PSF_EMA_ALPHA * new_psf[key])


# ─────────────────────────── Hungarian assignment ─────────────────────────────

def assign_blobs_hungarian(
    dots:  List[DotState],
    blobs: List[np.ndarray],
    gray:  np.ndarray,
) -> List[Optional[np.ndarray]]:
    """Globally optimal dot↔blob assignment via the Hungarian algorithm.

    Cost[i, j] = Euclidean distance from dot i's Kalman-predicted position to blob j
               + PSF_WEIGHT × PSF dissimilarity between dot i's reference and blob j.

    Assignments whose total cost exceeds MAX_ASSIGNMENT_COST are rejected so that
    the dot falls back to its Kalman prediction for that frame.
    """
    if not blobs:
        return [None] * len(dots)

    blob_psfs = [measure_psf(gray, b[0], b[1]) for b in blobs]

    n_d, n_b = len(dots), len(blobs)
    cost = np.full((n_d, n_b), 1e9)
    for i, dot in enumerate(dots):
        for j, blob in enumerate(blobs):
            dist     = float(np.linalg.norm(blob - dot.pos))
            psf_cost = PSF_WEIGHT * psf_dissimilarity(dot.psf_ref, blob_psfs[j])
            cost[i, j] = dist + psf_cost

    dot_idx, blob_idx = linear_sum_assignment(cost)
    assignments: List[Optional[np.ndarray]] = [None] * n_d
    for di, bj in zip(dot_idx, blob_idx):
        if cost[di, bj] < _max_assign_cost:
            assignments[di] = blobs[bj]

    return assignments


# ─────────────────────────── Centroid & detection ────────────────────────────

def circular_aperture_centroid(gray, x, y, radius=None):
    if radius is None: radius = _ap_r
    r  = int(np.ceil(radius))
    x0 = max(0, int(x) - r);  x1 = min(gray.shape[1], int(x) + r + 1)
    y0 = max(0, int(y) - r);  y1 = min(gray.shape[0], int(y) + r + 1)
    roi = gray[y0:y1, x0:x1].astype(np.float64)
    if roi.size == 0:
        return None
    # Estimate local background from the border ring of the ROI, then subtract it
    # so that the centroid is weighted by dot signal rather than background level.
    border = np.concatenate([roi[0, :], roi[-1, :], roi[1:-1, 0], roi[1:-1, -1]])
    bg = float(np.median(border)) if border.size else 0.0
    rows, cols = np.mgrid[y0:y1, x0:x1]
    mask    = ((cols - x) ** 2 + (rows - y) ** 2) <= radius ** 2
    weights = np.maximum(roi - bg, 0) * mask
    total   = weights.sum()
    if total == 0:
        return None
    cx = (weights * cols).sum() / total
    cy = (weights * rows).sum() / total
    return np.array([cx, cy]) if (np.isfinite(cx) and np.isfinite(cy)) else None


def blob_peak_brightness(gray, pos, radius=None):
    if radius is None: radius = _ap_r
    r  = int(np.ceil(radius))
    x0 = max(0, int(pos[0]) - r);  x1 = min(gray.shape[1], int(pos[0]) + r + 1)
    y0 = max(0, int(pos[1]) - r);  y1 = min(gray.shape[0], int(pos[1]) + r + 1)
    roi = gray[y0:y1, x0:x1]
    return float(roi.max()) if roi.size else 0.0


def detect_blobs(gray, thresh):
    """Detect bright spots by thresholding the top-hat-preprocessed image.

    Using contour-based detection on the contrast image rather than
    SimpleBlobDetector gives predictable behaviour for low-contrast dots on
    noisy, slowly-varying backgrounds.  `thresh` is in contrast units (ADU
    above local background), not absolute pixel values.
    """
    proc    = preprocess_frame(gray)
    blurred = cv2.GaussianBlur(proc, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
    _, binary = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if BLOB_MIN_AREA <= area <= BLOB_MAX_AREA:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0 and (4 * np.pi * area / perimeter ** 2) >= BLOB_MIN_CIRC:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    blobs.append(np.array([M['m10'] / M['m00'], M['m01'] / M['m00']]))
    return blobs


def find_bright_peaks(gray, n=2):
    """Find n brightest isolated spots in a single pass.

    Thresholds the top-hat image at a robust percentile of the peak brightness,
    then returns the top-n components by brightness. Simple, direct, no loops.
    """
    proc    = preprocess_frame(gray)
    blurred = cv2.GaussianBlur(proc, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
    max_val = int(blurred.max())
    if max_val < 5:
        return []

    # Single threshold at 50% of peak — typical for isolated peaks
    thresh_val = max(5, int(max_val * 0.50))
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

    if not candidates:
        return []

    # Sort by brightness and return top-n, sorted by x-position (left-to-right)
    candidates.sort(key=lambda t: t[0], reverse=True)
    peaks = [pos for _, pos in candidates[:n]]
    peaks.sort(key=lambda p: float(p[0]))
    return peaks


def find_brightest_near(gray: np.ndarray,
                        center: np.ndarray,
                        radius: float) -> Optional[np.ndarray]:
    """Centroid of the brightest blob within radius of center.

    Operates on the top-hat preprocessed image so it works on low-contrast dots.
    """
    proc    = preprocess_frame(gray)
    blurred = cv2.GaussianBlur(proc, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
    cx, cy  = int(round(float(center[0]))), int(round(float(center[1])))
    r       = int(np.ceil(radius))
    x0 = max(0, cx - r);  x1 = min(gray.shape[1], cx + r + 1)
    y0 = max(0, cy - r);  y1 = min(gray.shape[0], cy + r + 1)
    roi = blurred[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    peak_val = int(roi.max())
    if peak_val < 3:
        return None
    _, binary = cv2.threshold(roi, max(3, int(peak_val * 0.50)), 255, cv2.THRESH_BINARY)
    num_labels, _labels, _stats, centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    best_brightness, best_pos = -1.0, None
    for i in range(1, num_labels):
        lx = x0 + centroids[i][0]
        ly = y0 + centroids[i][1]
        b  = blob_peak_brightness(blurred, np.array([lx, ly]))
        if b > best_brightness:
            best_brightness = b
            best_pos = np.array([lx, ly])
    if best_pos is None:
        return None
    refined = circular_aperture_centroid(gray, best_pos[0], best_pos[1])
    return refined if refined is not None else best_pos


def find_bright_peaks_near(gray: np.ndarray,
                           center: np.ndarray,
                           radius: float,
                           n: int = 2) -> List[np.ndarray]:
    """Run find_bright_peaks on a cropped ROI; return positions in full-frame coords."""
    cx, cy = int(round(float(center[0]))), int(round(float(center[1])))
    r      = int(np.ceil(radius))
    x0 = max(0, cx - r);  x1 = min(gray.shape[1], cx + r + 1)
    y0 = max(0, cy - r);  y1 = min(gray.shape[0], cy + r + 1)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return []
    peaks_roi = find_bright_peaks(roi, n=n)
    return [p + np.array([x0, y0]) for p in peaks_roi]


def _ring_intensity_centroid(blurred: np.ndarray,
                             cx: float, cy: float,
                             r_inner: float, r_outer: float,
                             x0: int, y0: int) -> Optional[np.ndarray]:
    """Intensity-weighted centroid of pixels in the annulus [r_inner, r_outer)."""
    h, w = blurred.shape
    cols = np.arange(x0, x0 + w)
    rows = np.arange(y0, y0 + h)
    cols_g, rows_g = np.meshgrid(cols, rows)
    dist2 = (cols_g - cx) ** 2 + (rows_g - cy) ** 2
    mask = (dist2 >= r_inner ** 2) & (dist2 < r_outer ** 2)
    w_vals = blurred[mask].astype(np.float64)
    total  = w_vals.sum()
    if total < 1e-6:
        return None
    rx = float((cols_g[mask] * w_vals).sum() / total)
    ry = float((rows_g[mask] * w_vals).sum() / total)
    return np.array([rx, ry])


def estimate_intersection_positions(
        gray: np.ndarray,
        dots: List) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Resolve two Airy disks within a merged blob.

    Uses the Kalman-predicted separation axis directly — much more reliable than
    ring asymmetry analysis, especially on low-contrast backgrounds — then scans
    the top-hat-preprocessed image along that axis to find the two sub-peaks.
    The Kalman predictions also serve as positional priors so the correct peak
    pair is chosen even when SNR is low.

    Returns (pos_a, pos_b) in full-frame pixel coords, or (None, None) on failure.
    """
    delta = dots[1].pos - dots[0].pos
    pred_sep = float(np.linalg.norm(delta))
    if pred_sep < 1.0:
        return None, None

    axis_norm = delta / pred_sep
    midpoint  = (dots[0].pos + dots[1].pos) * 0.5
    cx, cy    = float(midpoint[0]), float(midpoint[1])

    # Top-hat image gives contrast values instead of raw ADU — essential on a
    # slowly-varying bright background where raw profiles have poor SNR.
    proc    = preprocess_frame(gray)
    blurred = cv2.GaussianBlur(proc, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0).astype(np.float64)

    # Scan range: at least the predicted separation, capped at _intersect_sr
    half_scan = max(pred_sep * 0.7, _ap_r * 2.0)
    n_steps   = max(60, int(half_scan * 4))
    ts        = np.linspace(-half_scan, half_scan, n_steps)
    profile   = np.zeros(n_steps)
    for k, t in enumerate(ts):
        px = int(round(cx + axis_norm[0] * t))
        py = int(round(cy + axis_norm[1] * t))
        if 0 <= px < blurred.shape[1] and 0 <= py < blurred.shape[0]:
            profile[k] = blurred[py, px]

    if profile.max() < 3.0:
        return None, None

    # Expected peak positions (Kalman prediction projected onto axis)
    t_a = float(np.dot(dots[0].pos - midpoint, axis_norm))
    t_b = float(np.dot(dots[1].pos - midpoint, axis_norm))

    # Find all local maxima above 35% of profile peak
    threshold = profile.max() * 0.35
    peak_candidates: List[Tuple[float, int, float]] = []
    for k in range(1, n_steps - 1):
        if (profile[k] > profile[k - 1] and profile[k] > profile[k + 1]
                and profile[k] >= threshold):
            t_peak = float(ts[k])
            # Score combines brightness and proximity to a Kalman-predicted position
            prox   = min(abs(t_peak - t_a), abs(t_peak - t_b))
            score  = profile[k] / (prox + pred_sep * 0.1)
            peak_candidates.append((score, k, t_peak))

    if len(peak_candidates) < 2:
        return None, None

    # Pick the best pair of peaks that are sufficiently separated
    peak_candidates.sort(reverse=True)
    min_pair_sep = pred_sep * 0.15
    best_t1 = best_t2 = None
    for i in range(len(peak_candidates)):
        for j in range(i + 1, len(peak_candidates)):
            t1 = peak_candidates[i][2]
            t2 = peak_candidates[j][2]
            if abs(t1 - t2) >= min_pair_sep:
                best_t1, best_t2 = (t1, t2) if t1 < t2 else (t2, t1)
                break
        if best_t1 is not None:
            break

    if best_t1 is None:
        return None, None

    pos_a = midpoint + axis_norm * best_t1
    pos_b = midpoint + axis_norm * best_t2

    # Assign peaks to dots by proximity to Kalman predictions
    if np.linalg.norm(pos_a - dots[1].pos) < np.linalg.norm(pos_a - dots[0].pos):
        pos_a, pos_b = pos_b, pos_a

    # Refine each position with a background-subtracted aperture centroid
    ra = circular_aperture_centroid(gray, pos_a[0], pos_a[1])
    rb = circular_aperture_centroid(gray, pos_b[0], pos_b[1])
    pos_a = ra if ra is not None else pos_a
    pos_b = rb if rb is not None else pos_b

    return pos_a, pos_b


def compute_radial_profile(gray: np.ndarray, cx: float, cy: float,
                           max_r: float) -> np.ndarray:
    """Mean intensity in 1-px-wide annuli from r=0 to max_r around (cx, cy)."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float64)
    r_max = int(np.ceil(max_r))
    x0 = max(0, int(cx) - r_max);  x1 = min(gray.shape[1], int(cx) + r_max + 1)
    y0 = max(0, int(cy) - r_max);  y1 = min(gray.shape[0], int(cy) + r_max + 1)
    patch     = blurred[y0:y1, x0:x1]
    cols_g, rows_g = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    dist_map  = np.sqrt((cols_g - cx) ** 2 + (rows_g - cy) ** 2)
    profile   = np.zeros(r_max + 1)
    for r in range(r_max + 1):
        mask = (dist_map >= max(0.0, r - 0.5)) & (dist_map < r + 0.5)
        if mask.sum() > 0:
            profile[r] = float(patch[mask].mean())
    return profile


def ring_edges_from_profile(profile: np.ndarray,
                            n_interior: int = 3) -> List[float]:
    """Find ring boundaries at the radii of steepest brightness change.

    Smooths the radial profile, computes |dI/dr|, then picks the n_interior
    largest local gradient peaks as interior ring boundaries.  Returns
    [0, r1, r2, …, max_r] — always at least 2 elements (0 and max_r).
    """
    if len(profile) < 5:
        return [0.0, float(len(profile) - 1)]

    smoothed = np.convolve(profile.astype(float), np.ones(3) / 3.0, mode='same')
    grad     = np.abs(np.gradient(smoothed))
    min_height = grad.max() * 0.10   # ignore tiny ripples

    peaks = []
    for i in range(1, len(grad) - 1):
        if (grad[i] > grad[i - 1] and grad[i] > grad[i + 1]
                and grad[i] >= min_height):
            peaks.append((grad[i], i))

    peaks.sort(reverse=True)
    interior = sorted(float(i) for _, i in peaks[:n_interior])
    return [0.0] + interior + [float(len(profile) - 1)]


def calibrate_tracking_params(gray: np.ndarray, dots: list) -> None:
    """Derive tracking constants from the initial dots' PSF.

    Measures FWHM of both dots and scales all radii/distances to match the optics.
    Also initializes Kalman velocity from the current separation vector.
    """
    global _ap_r, _bg_inner_r, _bg_outer_r
    global _intersect_dist, _intersect_sr, _ring_edges_d, _max_assign_cost

    fwhms = []
    for d in dots:
        if d is None:
            continue
        psf = measure_psf(gray, d.pos[0], d.pos[1],
                          sig_r=APERTURE_RADIUS,
                          bg_inner=BG_INNER_RADIUS,
                          bg_outer=BG_OUTER_RADIUS)
        fw = psf.get('fwhm_mean', 0.0)
        if fw > 2.0:
            fwhms.append(fw)
            d.psf_ref = {k: psf[k] for k in ('peak', 'flux', 'fwhm_mean')}

    if not fwhms:
        print("[calibrate] PSF measurement failed — keeping defaults")
        return

    avg_fwhm = float(np.mean(fwhms))

    # Aperture: ~2.5× FWHM to capture the disk core plus noise (Airy disk total)
    ap_r = max(avg_fwhm * 2.5, 10.0)

    _ap_r            = ap_r
    _bg_inner_r      = ap_r * 1.17
    _bg_outer_r      = ap_r * 1.50
    _intersect_dist  = ap_r * 1.3
    _intersect_sr    = ap_r * 2.0
    _max_assign_cost = ap_r * 8.0

    # Simple ring edges: linearly scaled from aperture radius
    _ring_edges_d = [0.0, ap_r * 0.20, ap_r * 0.40, ap_r * 0.65, ap_r]

    # Initialize Kalman velocity from current separation (birds-eye estimate of speed)
    if len(dots) >= 2 and dots[0] is not None and dots[1] is not None:
        sep_vec = dots[1].pos - dots[0].pos
        for d in dots:
            d.kf_x[2:4] = sep_vec / 20.0   # assume separation takes ~20 frames to cover

    print(f"[calibrate] fwhm={avg_fwhm:.1f}px aperture={ap_r:.1f}px "
          f"intersect_dist={_intersect_dist:.1f}px")


def auto_init_dots(gray, thresh):
    """Seed Dot A/B from the two brightest spots.

    Uses top-hat preprocessing + connected components. Dots sorted left→right.
    Returns (pos_a, pos_b) or (None, None) on failure.
    """
    peaks = find_bright_peaks(gray, n=2)
    if len(peaks) >= 2:
        return peaks[0], peaks[1]
    if len(peaks) == 1:
        return peaks[0], None
    return None, None


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
    frame_us = int(1_000_000 / TARGET_FPS)   # microseconds per frame at target fps
    config = picam2.create_video_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
        controls={
            "FrameDurationLimits": (frame_us, frame_us),  # locks fps exactly
            "ExposureTime": frame_us,                     # full-frame exposure at 60 fps
            "AnalogueGain": 2.0,                          # 2× sensor gain; raise if dots are dim
            "AeEnable": False,                            # disable AEC so exposure stays fixed
        }
    )
    picam2.configure(config)
    picam2.start()
    sleep(0.5)

    dot_a = dot_b = None
    print("Waiting for initial blobs…")
    for _ in range(60):
        rgb  = picam2.capture_array()
        gray = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        pa, pb = auto_init_dots(gray, _thresh)
        if pa is not None and pb is not None:
            dot_a = DotState("Dot A", pa)
            dot_b = DotState("Dot B", pb)
            dot_a.trail.append(tuple(pa.astype(int)))
            dot_b.trail.append(tuple(pb.astype(int)))
            calibrate_tracking_params(gray, [dot_a, dot_b])
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
                dots[0].trail.append(tuple(pa.astype(int)))
            if pb is not None:
                dots[1] = DotState("Dot B", pb)
                dots[1].trail.append(tuple(pb.astype(int)))
            if pa is not None and pb is not None:
                calibrate_tracking_params(gray, dots)
            _needs_reset = False

        # Kalman predict every frame regardless of separation
        for dot in dots:
            dot.kf_predict()

        # Blob detection
        raw = detect_blobs(gray, _thresh)
        refined = []
        for b in raw:
            c = circular_aperture_centroid(gray, b[0], b[1])
            refined.append(c if c is not None else b)

        sep  = float(np.linalg.norm(dots[0].pos - dots[1].pos))
        near = sep < _intersect_dist   # dots close enough that blobs may have merged

        if len(refined) >= 2 or (len(refined) == 1 and not near):
            # Two (or more) distinct blobs, or one blob with dots well-separated:
            # standard Hungarian assignment works fine.
            assignments = assign_blobs_hungarian(dots, refined, gray)
            for dot, matched_pos in zip(dots, assignments):
                if matched_pos is not None:
                    dot.kf_update(matched_pos)
                else:
                    dot.mark_lost()

        elif len(refined) == 1 and near:
            # Single merged blob: try to resolve two sub-peaks using the
            # Kalman-predicted separation axis + top-hat brightness profile.
            pos_a, pos_b = estimate_intersection_positions(gray, dots)
            if pos_a is not None and pos_b is not None:
                dots[0].kf_update(pos_a)
                dots[1].kf_update(pos_b)
            else:
                # Can't resolve — both dots coast at merged centroid
                midpoint = (dots[0].pos + dots[1].pos) * 0.5
                merged_pos = find_brightest_near(gray, midpoint, _intersect_sr)
                for dot in dots:
                    dot.freeze(merged_pos if merged_pos is not None else refined[0])

        else:  # zero blobs detected
            for dot in dots:
                dot.mark_lost()

        sep          = float(np.linalg.norm(dots[0].pos - dots[1].pos))
        intersecting = sep < _intersect_dist

        # Step 2: Measure PSF; update identity reference only for confirmed tracks
        for dot in dots:
            dot.psf = measure_psf(gray, dot.pos[0], dot.pos[1])
            if not dot.predicted:
                update_psf_ref(dot, dot.psf)

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
