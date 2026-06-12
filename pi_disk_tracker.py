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

Dependencies: flask, flask-sock, picamera2, cv2, numpy, scipy, requests
"""

from picamera2 import Picamera2
import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from time import sleep, perf_counter
import threading
import random

import requests as _requests

from flask import Flask, render_template, jsonify, Response, request
from flask_sock import Sock
from scipy.optimize import linear_sum_assignment, least_squares

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

# Ellipticity of the merged blob below which the two Airy disks are considered
# well-aligned (PSF is near-circular).  At this point both dots are reported at
# the merged peak centroid rather than frozen.
ALIGNED_ELLIPTICITY_THRESH = 0.20

# ── Servo / inter-Pi config ───────────────────────────────────────────────────

SERVO_PI_URL          = "http://172.20.10.3:5000"   # set to slave Pi IP if mDNS not available
AUTO_ALIGN_SCORE_TARGET = 0.92   # alignment_score above which auto-align considers done
AUTO_ALIGN_SETTLE     = 0.35     # seconds between jog and next measurement

# ── "Drive dot to target" controller (3 servos, one dot) ──────────────────────
# All three servos move the same dot, each along its own direction in the image.
# The controller learns a 2×3 image Jacobian J — column i is the dot's pixel
# displacement per +jog of servo i — then drives the 2-D target error to zero
# with the least-norm jog vector  n = Kp · J⁺ · e  (J⁺ = pseudo-inverse).
# J is bootstrapped by probing each servo once and refined online (LMS) as it
# runs, so it tracks drift and changing mirror response without recalibration.
ALIGN_SERVOS      = ("A", "B", "C") # servos driving the dot, in Jacobian-column order
ALIGN_TOL_PX      = 4.0   # stop correcting once dot is within this of the target
ALIGN_LOOP_DT     = 0.08  # continuous control period — servos keep running between updates
ALIGN_MAX_SPEED   = 0.40  # cap on per-servo velocity command (offset from STOP)
ALIGN_PROBE_SPEED = 0.18  # velocity used to identify each servo's effect
ALIGN_PROBE_TMAX  = 1.0   # max seconds to run a probe before recording its column
ALIGN_MIN_DISP    = 1.5   # px — require this much travel to trust a probe measurement
ALIGN_LMS_RATE    = 0.3   # online Jacobian LMS update rate
# LQR weights for the integrator  x_{k+1} = x + (G·dt)·w  (x = pos−target, w = velocities,
# G = velocity Jacobian: px/s of dot motion per unit servo velocity command)
ALIGN_Q           = (1.0, 1.0) # diagonal of the 2×2 position-error weight Q
ALIGN_R           = 0.05       # control-effort weight → R = ALIGN_R·I₃
                               # main tuning knob: larger = gentler/slower motion

# ── LQR model ─────────────────────────────────────────────────────────────────
# State:   x = [e_x, e_y, ė_x, ė_y]  midpoint error (px) and rate (px/frame)
# Control: u = [u_tip, u_tilt]        continuous servo command (jog units)
# dt = AUTO_ALIGN_SETTLE; Euler-discretised damped integrator per axis
#
#   A_d = diag-block([[1, dt], [0, 1-β·dt]])   β = velocity decay / s
#   B_d = diag-block([[0], [k·dt]])             k = px/s per unit command
#
_LQR_dt   = AUTO_ALIGN_SETTLE
_LQR_beta = 2.0     # tune to match mirror damping
_LQR_k    = 8.0     # px / s per unit servo command — tune to optics

def _build_lqr_matrices():
    dt = _LQR_dt
    a22 = 1.0 - _LQR_beta * dt
    A = np.array([[1.0, 0.0,       dt, 0.0     ],
                  [0.0, 1.0,      0.0,       dt],
                  [0.0, 0.0,     a22, 0.0     ],
                  [0.0, 0.0,     0.0,      a22]])
    B = np.array([[0.0,         0.0        ],
                  [0.0,         0.0        ],
                  [_LQR_k * dt, 0.0        ],
                  [0.0,         _LQR_k * dt]])
    Q = np.diag([1.0, 1.0, 0.1, 0.1])   # penalise position errors
    R = np.diag([0.01, 0.01])            # allow aggressive control
    return A, B, Q, R

_LQR_A, _LQR_B, _LQR_Q, _LQR_R = _build_lqr_matrices()

try:
    from scipy.linalg import solve_discrete_are as _dare_solve
    _HAVE_DARE = True
except Exception as _dare_err:
    print(f"[WARN] scipy DARE unavailable ({_dare_err}) — LQR uses damped-inverse fallback")
    _HAVE_DARE = False

try:
    _P     = _dare_solve(_LQR_A, _LQR_B, _LQR_Q, _LQR_R)
    _LQR_K = np.linalg.inv(_LQR_R + _LQR_B.T @ _P @ _LQR_B) @ (_LQR_B.T @ _P @ _LQR_A)
    print(f"LQR gain K:\n{_LQR_K.round(4)}")
except Exception as _lqr_err:
    print(f"[WARN] LQR solve failed: {_lqr_err}")
    _LQR_K = np.zeros((2, 4))

# ── Runtime-calibrated physics parameters (overwritten by calibrate_tracking_params) ──
_ap_r            = float(APERTURE_RADIUS)
_bg_inner_r      = float(BG_INNER_RADIUS)
_bg_outer_r      = float(BG_OUTER_RADIUS)
_intersect_dist  = float(INTERSECTION_DIST)
_intersect_sr    = float(INTERSECTION_SEARCH_RADIUS)
_ring_edges_d    = list(RING_EDGES)
_max_assign_cost = float(APERTURE_RADIUS * 8)
_fwhm_calib      = float(APERTURE_RADIUS) / 2.5 * 2.3548   # bootstrap; updated by calibrate

# Sparrow limit: below ~0.84 × FWHM the two-Gaussian model is statistically
# indistinguishable from a single Gaussian, so don't try to resolve sub-dots.
SPARROW_FWHM_FRAC = 0.84

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
_overlay_on     = True
_auto_align     = False
_servo_online   = False

# Camera exposure state + auto-tune control (auto-tune runs in the capture thread)
CAM_GAIN_MIN    = 1.0     # AnalogueGain rails
CAM_GAIN_MAX    = 16.0
CAM_TARGET_PEAK = 235.0   # aim brightest dot pixels here — bright but below 255 clip
CAM_PEAK_TOL    = 12.0    # px-value band around the target that counts as converged
# Exposure range (µs): the FrameDurationLimits lock caps it just under the frame
# period, so the manual slider can span from very short to ~one frame budget.
CAM_EXP_MIN     = 100
CAM_EXP_MAX     = int(1_000_000 / TARGET_FPS) - 200
_cam_gain       = 8.0     # current AnalogueGain (raised/lowered by auto-tune)
_cam_exposure_us = None   # current ExposureTime, set in capture_loop
_cam_exposure_req = None  # pending ExposureTime from the UI, applied by capture thread
_autotune_req   = False   # set by /camera/auto, serviced by the capture thread
_autotune_busy  = False   # True while a tune is running

# Align controller state (3 servos → one dot)
_align_dot      = "A"                 # which dot label to drive ("A"/"B")
_target         = [FRAME_WIDTH / 2.0, FRAME_HEIGHT / 2.0]   # px target for the dot
_align_jac      = np.zeros((2, 3))    # image Jacobian: column i = px move per +jog of servo i
_align_known    = [False, False, False]   # whether each column has been identified

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


# ── Per-frame cache for top-hat + Gaussian-blurred image ─────────────────────
# Detection / resolution / merged-peak code paths can each ask for the blurred
# top-hat image. Without caching, a single near-intersection frame recomputes
# the (expensive) 41-px top-hat 4–5 times. Keyed on the gray buffer's memory
# address so we never serve a stale frame.
_blur_cache_key: Optional[int] = None
_blur_cache_proc: Optional[np.ndarray] = None
_blur_cache_blurred: Optional[np.ndarray] = None


def get_proc_blurred(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (top-hat, blurred top-hat) for `gray`, computing once per frame."""
    global _blur_cache_key, _blur_cache_proc, _blur_cache_blurred
    key = int(gray.ctypes.data)
    if _blur_cache_key != key or _blur_cache_proc is None:
        proc = preprocess_frame(gray)
        blurred = cv2.GaussianBlur(proc, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
        _blur_cache_key = key
        _blur_cache_proc = proc
        _blur_cache_blurred = blurred
    return _blur_cache_proc, _blur_cache_blurred


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


# ─────────────────────────── Intersection quality ────────────────────────────

def measure_intersection_quality(gray: np.ndarray,
                                  merged_pos: np.ndarray,
                                  dots: List[DotState]) -> Dict[str, Any]:
    """Alignment quality of two overlapping Airy disks.

    Two signals:
      ellipticity — PSF shape of the merged blob.  Approaches 0 when the disks
                    are perfectly co-centred (combined PSF is circular).
      peak_ratio  — measured peak vs. the expected peak at full incoherent overlap
                    (sum of per-dot PSF references).  Approaches 1 at alignment.

    The combined alignment_score in [0, 1] can be used directly as a control
    error signal: optimise toward 1.

    direction_deg is the angle of the PSF major axis (= the remaining separation
    axis).  A control loop should move one dot in this direction relative to the
    other to reduce misalignment.
    """
    psf = measure_psf(gray, float(merged_pos[0]), float(merged_pos[1]))
    if not psf:
        return {'alignment_score': 0.0, 'ellipticity': 1.0,
                'direction_deg': 0.0, 'peak': 0.0, 'peak_ratio': 0.0}

    ellipticity = float(psf.get('ellipticity', 1.0))
    peak        = float(psf.get('peak', 0.0))
    direction   = float(psf.get('angle_deg', 0.0))

    # Expected peak at perfect incoherent overlap = sum of individual PSF refs.
    # Fall back to shape-only score when references haven't been built yet.
    expected_peak = sum(d.psf_ref.get('peak', 0.0) for d in dots if d.psf_ref)
    if expected_peak > 1.0:
        peak_ratio      = float(np.clip(peak / expected_peak, 0.0, 1.5))
        alignment_score = float(np.clip(
            (1.0 - ellipticity) * min(peak_ratio, 1.0), 0.0, 1.0))
    else:
        peak_ratio      = 0.0
        alignment_score = float(np.clip(1.0 - ellipticity, 0.0, 1.0))

    return {
        'alignment_score': round(alignment_score, 3),
        'ellipticity':     round(ellipticity,     3),
        'direction_deg':   round(direction,        1),
        'peak':            round(peak,             1),
        'peak_ratio':      round(peak_ratio,       3),
    }


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
    _, blurred = get_proc_blurred(gray)
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
    """Find n brightest isolated spots using adaptive threshold + connected components.

    Operates on the top-hat preprocessed image so thresholds are in contrast
    units, making the search robust to absolute brightness changes.
    """
    _, blurred = get_proc_blurred(gray)
    max_val = int(blurred.max())
    if max_val < 5:
        return []

    candidates = []
    for pct in (0.75, 0.60, 0.45, 0.30, 0.20):
        thresh_val = max(5, int(max_val * pct))
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


def find_brightest_near(gray: np.ndarray,
                        center: np.ndarray,
                        radius: float) -> Optional[np.ndarray]:
    """Centroid of the brightest blob within radius of center.

    Operates on the top-hat preprocessed image so it works on low-contrast dots.
    """
    _, blurred = get_proc_blurred(gray)
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


def _get_separation_axis(gray: np.ndarray,
                         dots: List,
                         kalman_axis: np.ndarray) -> np.ndarray:
    """Blend the Kalman-predicted separation axis with the merged blob's ellipse fit.

    When two Airy disks form an elongated merged blob the major axis of a fitted
    ellipse is a direct, observation-driven estimate of the separation direction.
    Eccentricity is used as a confidence weight: near-circular blobs trust Kalman;
    elongated blobs trust the ellipse.
    """
    _, blurred = get_proc_blurred(gray)
    _, binary = cv2.threshold(blurred, _thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    midpoint  = (dots[0].pos + dots[1].pos) * 0.5
    best_cnt  = None
    best_dist = float('inf')
    for cnt in contours:
        if len(cnt) < 5:
            continue
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        d = float(np.hypot(M['m10'] / M['m00'] - midpoint[0],
                           M['m01'] / M['m00'] - midpoint[1]))
        if d < best_dist:
            best_dist = d
            best_cnt  = cnt

    if best_cnt is None or best_dist > _intersect_sr:
        return kalman_axis

    try:
        (_, _), (ea, eb), angle = cv2.fitEllipse(best_cnt)
        a, b = max(ea, eb) / 2.0, min(ea, eb) / 2.0
        if a < 1.0:
            return kalman_axis

        eccentricity = float(np.sqrt(max(0.0, 1.0 - (b / a) ** 2)))

        # OpenCV fitEllipse angle: degrees clockwise from the x-axis to major axis
        angle_rad   = np.radians(angle)
        ellipse_vec = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        if np.dot(ellipse_vec, kalman_axis) < 0:
            ellipse_vec = -ellipse_vec

        # Alpha ramps 0→1 as blob becomes more elongated
        alpha   = min(eccentricity * 1.5, 1.0)
        blended = (1.0 - alpha) * kalman_axis + alpha * ellipse_vec
        norm    = float(np.linalg.norm(blended))
        return blended / norm if norm > 1e-6 else kalman_axis
    except Exception:
        return kalman_axis


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
    # Below the Sparrow limit a 1-D scan cannot separate the peaks either —
    # let the merged-peak fallback handle it.
    if pred_sep < max(_fwhm_calib * SPARROW_FWHM_FRAC, 2.0):
        return None, None

    # Use blob ellipse fit to improve the axis estimate when the merged blob
    # is elongated; falls back to pure Kalman axis for near-circular blobs.
    axis_norm = _get_separation_axis(gray, dots, delta / pred_sep)
    midpoint  = (dots[0].pos + dots[1].pos) * 0.5
    cx, cy    = float(midpoint[0]), float(midpoint[1])

    # Top-hat image gives contrast values instead of raw ADU — essential on a
    # slowly-varying bright background where raw profiles have poor SNR.
    _, blurred_u8 = get_proc_blurred(gray)
    blurred = blurred_u8.astype(np.float64)

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


def fit_double_gaussian(gray: np.ndarray,
                        dots: List) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Fit two 2D Gaussians to the merged blob with three hard constraints:

    1. σ is locked to the calibrated PSF width (same optics → same width).
    2. Amplitude ratio r = peak_b / peak_a is locked to the EMA-tracked PSF refs.
    3. The flux-weighted midpoint is anchored to the merged blob's centroid,
       which is a high-SNR measurement that costs almost nothing.

    This collapses the original 8 free parameters (2 positions, 2 amplitudes,
    σ, bg) down to 4: a 2-D separation vector around the centroid, one
    amplitude, and the background.  With far fewer degrees of freedom the fit:
      • stays stable much closer to the Sparrow limit,
      • converges in a fraction of the iterations,
      • can't drift into a degenerate single-Gaussian solution.

    Returns (pos_a, pos_b) in full-frame pixel coords, or (None, None) on failure.
    """
    pred_sep = float(np.linalg.norm(dots[1].pos - dots[0].pos))

    # Below the Sparrow limit the two-Gaussian model is statistically
    # indistinguishable from a single Gaussian — let the aligned-merged-peak
    # branch handle it instead of producing a noisy split.
    sparrow_sep = max(_fwhm_calib * SPARROW_FWHM_FRAC, 2.0)
    if pred_sep < sparrow_sep:
        return None, None

    midpoint_kf = (dots[0].pos + dots[1].pos) * 0.5

    roi_r = max(int(np.ceil(pred_sep * 0.65 + _ap_r * 1.5)), int(np.ceil(_intersect_sr)))
    cx = int(round(float(midpoint_kf[0])))
    cy = int(round(float(midpoint_kf[1])))
    x0 = max(0, cx - roi_r);  x1 = min(gray.shape[1], cx + roi_r + 1)
    y0 = max(0, cy - roi_r);  y1 = min(gray.shape[0], cy + roi_r + 1)

    _, blurred_u8 = get_proc_blurred(gray)
    roi = blurred_u8[y0:y1, x0:x1].astype(np.float64)

    if roi.size == 0 or roi.max() < 3.0:
        return None, None

    cols_g = np.arange(x0, x1, dtype=np.float64)
    rows_g = np.arange(y0, y1, dtype=np.float64)
    X, Y   = np.meshgrid(cols_g, rows_g)
    data   = roi.ravel()

    # Locked PSF width
    sigma = max(_fwhm_calib / 2.3548, 1.0)
    inv_2sig2 = 1.0 / (2.0 * sigma * sigma)

    # Locked amplitude ratio r = amp_b / amp_a from EMA-tracked PSF refs.
    pa = float(dots[0].psf_ref.get('peak', 0.0))
    pb = float(dots[1].psf_ref.get('peak', 0.0))
    r = (pb / pa) if (pa > 1.0 and pb > 1.0) else 1.0

    # Flux-weighted centroid (background-subtracted) — anchors the midpoint.
    # With locked amp ratio r, the flux centroid lies at
    #   c = (pos_a + r·pos_b) / (1 + r)
    # so we parameterise the two positions as
    #   pos_a = c - (r/(1+r)) · s
    #   pos_b = c + (1/(1+r)) · s
    # leaving just the separation vector s = (sx, sy) and amplitude/bg free.
    bg0 = float(np.percentile(roi, 15))
    roi_sub = np.maximum(roi - bg0, 0.0)
    total = roi_sub.sum()
    if total < 1e-6:
        return None, None
    cx_flux = float((roi_sub * X).sum() / total)
    cy_flux = float((roi_sub * Y).sum() / total)

    fa = r / (1.0 + r)   # weight applied to s when subtracted from c → pos_a
    fb = 1.0 / (1.0 + r) # weight applied to s when added to c → pos_b

    # Initial separation vector (Kalman-seeded).
    s0 = dots[1].pos - dots[0].pos
    amp0 = float(roi.max() - bg0) * 0.9
    amp0 = max(amp0, 1.0)

    p0 = [float(s0[0]), float(s0[1]), amp0, bg0]
    sep_bound = pred_sep * 3.0 + sigma * 6.0
    lo = [-sep_bound, -sep_bound, 0.0, 0.0]
    hi = [ sep_bound,  sep_bound, float(roi.max() * 2.0), float(roi.max())]

    def residuals(p):
        sx, sy, a, bg = p
        gx1 = cx_flux - fa * sx
        gy1 = cy_flux - fa * sy
        gx2 = cx_flux + fb * sx
        gy2 = cy_flux + fb * sy
        g1 = a       * np.exp(-((X - gx1) ** 2 + (Y - gy1) ** 2) * inv_2sig2)
        g2 = (a * r) * np.exp(-((X - gx2) ** 2 + (Y - gy2) ** 2) * inv_2sig2)
        return (bg + g1 + g2).ravel() - data

    try:
        result = least_squares(residuals, p0, bounds=(lo, hi),
                               max_nfev=100, ftol=1e-4, xtol=1e-4)
        if result.status < 1:
            return None, None

        sx, sy, a, _ = result.x
        fit_sep = float(np.hypot(sx, sy))

        # Reject degenerate fits.
        if a < 1.0:
            return None, None
        if fit_sep < sparrow_sep or fit_sep > pred_sep * 3.0:
            return None, None

        pos_a = np.array([cx_flux - fa * sx, cy_flux - fa * sy])
        pos_b = np.array([cx_flux + fb * sx, cy_flux + fb * sy])

        # Assign by proximity to Kalman predictions
        if np.linalg.norm(pos_a - dots[1].pos) < np.linalg.norm(pos_a - dots[0].pos):
            pos_a, pos_b = pos_b, pos_a

        return pos_a, pos_b
    except Exception:
        return None, None


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
    """Derive all physics-based tracking constants from the initial dots' PSF.

    Uses the default aperture for a bootstrap PSF measurement, then scales every
    radius/distance parameter from the measured average FWHM so the tracker
    adapts automatically to whatever optics are in use.
    """
    global _ap_r, _bg_inner_r, _bg_outer_r
    global _intersect_dist, _intersect_sr, _ring_edges_d, _max_assign_cost
    global _fwhm_calib

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

    if not fwhms:
        print("[calibrate] PSF measurement failed — keeping default parameters")
        return

    avg_fwhm = float(np.mean(fwhms))

    # Aperture: capture the central disk (~2.5× FWHM), minimum 10 px
    ap_r = max(avg_fwhm * 2.5, 10.0)

    _fwhm_calib      = avg_fwhm
    _ap_r            = ap_r
    _bg_inner_r      = ap_r * 1.17           # just outside aperture
    _bg_outer_r      = ap_r * 1.50           # background annulus outer edge
    _intersect_dist  = ap_r * 1.3            # blobs start merging at ~1 diameter separation
    _intersect_sr    = ap_r * 2.0            # search radius for merged blob centroid
    _max_assign_cost = ap_r * 8.0            # Hungarian rejection threshold

    # Ring edges: derived from radii of steepest brightness gradient, averaged
    # across both dots so the rings adapt to the actual PSF structure.
    all_edges: List[List[float]] = []
    for d in dots:
        if d is None:
            continue
        prof  = compute_radial_profile(gray, d.pos[0], d.pos[1], ap_r)
        edges = ring_edges_from_profile(prof, n_interior=3)
        if len(edges) >= 3:
            all_edges.append(edges)

    if all_edges:
        # Element-wise mean; handle the case where edge counts differ
        min_len = min(len(e) for e in all_edges)
        _ring_edges_d = [
            float(np.mean([e[i] for e in all_edges]))
            for i in range(min_len)
        ]
        # Ensure last edge reaches the aperture boundary
        if _ring_edges_d[-1] < ap_r * 0.9:
            _ring_edges_d.append(ap_r)
    else:
        _ring_edges_d = [0.0, ap_r * 0.20, ap_r * 0.40, ap_r * 0.65, ap_r]

    edges_str = ", ".join(f"{e:.1f}" for e in _ring_edges_d)
    print(f"[calibrate] fwhm={avg_fwhm:.1f}px  aperture={ap_r:.1f}px  "
          f"intersect_dist={_intersect_dist:.1f}px  "
          f"ring_edges=[{edges_str}]")


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

def _grab_gray(picam2) -> np.ndarray:
    """Capture one frame and return it as greyscale (same pipeline as the loop)."""
    rgb = picam2.capture_array()
    return cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)


def _auto_tune_camera(picam2) -> None:
    """Adjust the camera for the best dot-acquisition conditions.

    The dots are bright spots on a dark field, so the ideal exposure puts the
    brightest dot pixels just below saturation: bright enough for good SNR, but
    not clipped (clipping flattens the PSF and ruins centroiding).  We servo the
    AnalogueGain so the dot-core brightness lands near CAM_TARGET_PEAK, then set
    the detection threshold partway between the background and the dot peak and
    re-seed the dots.
    """
    global _cam_gain, _thresh, _needs_reset

    def _dot_peak(g):
        # Peak of a 3×3-median-filtered frame: preserves the true dot-core
        # brightness at any dot size (so real clipping is detected) while
        # rejecting single hot pixels.
        return float(cv2.medianBlur(g, 3).max())

    gain = float(_cam_gain)
    gray = _grab_gray(picam2)
    for _ in range(10):
        peak = _dot_peak(gray)
        if abs(peak - CAM_TARGET_PEAK) <= CAM_PEAK_TOL:
            break
        if peak >= 253.0:
            ratio = 0.5                     # clipping — back off hard to escape saturation
        else:
            ratio = float(np.clip(CAM_TARGET_PEAK / max(peak, 1.0), 0.5, 2.0))
        new_gain = float(np.clip(gain * ratio, CAM_GAIN_MIN, CAM_GAIN_MAX))
        if abs(new_gain - gain) < 1e-3:
            break   # hit a gain rail — can't improve further
        gain = new_gain
        picam2.set_controls({"AnalogueGain": gain})
        sleep(0.25)   # let the sensor apply the new gain
        gray = _grab_gray(picam2)

    _cam_gain = gain

    # Detection threshold: partway between background level and the dot peak.
    bg   = float(np.median(gray))
    peak = _dot_peak(gray)
    _thresh = int(np.clip(bg + 0.45 * (peak - bg), 20, 250))
    _needs_reset = True   # re-acquire the dots under the new exposure/threshold
    print(f"[camera] auto-tune → gain={_cam_gain:.2f}  peak={peak:.0f}  "
          f"bg={bg:.0f}  thresh={_thresh}")


def capture_loop():
    global _jpeg_frame, _tracking_state, _thresh, _frame_idx
    global _fps_actual, _intersecting, _needs_reset
    global _cam_exposure_us, _cam_exposure_req, _autotune_req, _autotune_busy

    picam2 = Picamera2()
    frame_us = int(1_000_000 / TARGET_FPS)   # microseconds per frame at target fps
    # Keep frame-format/size at config time, but apply *exposure* controls after
    # start(). picamera2 clamps ExposureTime/AnalogueGain set in the config block
    # to sensor minimums before the modes are known, which produces a black frame.
    config = picam2.create_video_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
        controls={
            "FrameDurationLimits": (frame_us, frame_us),  # locks fps exactly
        },
    )
    picam2.configure(config)
    picam2.start()
    sleep(0.5)

    # Fixed manual exposure for stable dot brightness (PSF identity matching needs
    # it). Applied after start() so the sensor honours the values. ExposureTime is
    # kept just under the 60-fps frame budget; AnalogueGain carries the brightness.
    _cam_exposure_us = frame_us - 200    # ~16.5 ms, just under the frame period
    picam2.set_controls({
        "AeEnable":     False,            # disable AEC so exposure stays fixed
        "AwbEnable":    False,            # disable auto white balance
        "ExposureTime": _cam_exposure_us,
        "AnalogueGain": _cam_gain,        # auto-tuned by /camera/auto (dim↑ / washed-out↓)
        "ColourGains":  (1.0, 1.0),      # neutral R/B gains (no colour tint)
        "Saturation":   0.0,             # strip colour → true greyscale
        "Contrast":     2.0,             # boost luminance contrast
        "Sharpness":    2.0,             # sharpen edges for crisp dot boundaries
    })
    sleep(0.3)   # let the new exposure settle before grabbing init frames

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
        # Service a camera auto-tune request in this thread (it owns picam2).
        if _autotune_req:
            _autotune_busy = True
            try:
                _auto_tune_camera(picam2)
            except Exception as _cam_e:
                print(f"[camera] auto-tune failed: {_cam_e}")
            _autotune_req  = False
            _autotune_busy = False

        # Apply a pending manual exposure change from the UI slider.
        if _cam_exposure_req is not None:
            exp = int(max(CAM_EXP_MIN, min(CAM_EXP_MAX, _cam_exposure_req)))
            try:
                picam2.set_controls({"ExposureTime": exp})
                _cam_exposure_us = exp
            except Exception as _exp_e:
                print(f"[camera] exposure set failed: {_exp_e}")
            _cam_exposure_req = None

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

        elif near:
            # Merged or absent blob — try sub-dot resolution in order of accuracy.

            # 1. Double-Gaussian fit directly to ROI pixel data (most accurate;
            #    works until centres are within ~1 FWHM of each other).
            pos_a, pos_b = fit_double_gaussian(gray, dots)

            # 2. 1-D profile scan along the ellipse-guided separation axis.
            if pos_a is None:
                pos_a, pos_b = estimate_intersection_positions(gray, dots)

            if pos_a is not None and pos_b is not None:
                dots[0].kf_update(pos_a)
                dots[1].kf_update(pos_b)
            else:
                # Full merge — individual peaks unresolvable.
                # Use PSF shape and brightness to decide: if the merged blob is
                # near-circular (ellipticity < threshold) the dots are well
                # aligned — track the merged peak normally rather than freezing.
                midpoint   = (dots[0].pos + dots[1].pos) * 0.5
                merged_pos = find_brightest_near(gray, midpoint, _intersect_sr)
                if merged_pos is None and refined:
                    merged_pos = refined[0]
                if merged_pos is not None:
                    iq = measure_intersection_quality(gray, merged_pos, dots)
                    if iq['ellipticity'] < ALIGNED_ELLIPTICITY_THRESH:
                        # Dots are well-aligned — both update to merged peak.
                        for dot in dots:
                            dot.kf_update(merged_pos)
                    else:
                        for dot in dots:
                            dot.freeze(merged_pos)
                else:
                    for dot in dots:
                        dot.freeze(None)

        else:  # zero blobs, dots not near — both lost
            for dot in dots:
                dot.mark_lost()

        sep          = float(np.linalg.norm(dots[0].pos - dots[1].pos))
        intersecting = sep < _intersect_dist

        # Step 2: Measure PSF; update identity reference only for confirmed tracks
        for dot in dots:
            dot.psf = measure_psf(gray, dot.pos[0], dot.pos[1])
            if not dot.predicted:
                update_psf_ref(dot, dot.psf)

        frame_out = (draw_overlay(gray, dots, intersecting)
                     if _overlay_on else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
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
        if intersecting:
            merged_mid = (dots[0].pos + dots[1].pos) * 0.5
            state['intersection_quality'] = measure_intersection_quality(
                gray, merged_mid, dots)
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


# ─────────────────────────── Auto-alignment loop ─────────────────────────────

def _servo_post(cmd: str) -> bool:
    """Send one discrete servo command (e.g. 'a+', 'tip-') to the slave Pi."""
    global _servo_online
    for _attempt in range(2):
        try:
            r = _requests.post(f"{SERVO_PI_URL}/servo",
                               json={"command": cmd}, timeout=2.0)
            _servo_online = r.ok
            return r.ok
        except Exception:
            if _attempt == 0:
                sleep(0.06)
    _servo_online = False
    return False


def _servo_velocity(w) -> bool:
    """Stream continuous per-servo velocities to the slave (non-blocking).

    `w` is a 3-vector of signed velocity commands (offset from STOP) aligned
    with ALIGN_SERVOS.  The servos keep running at these velocities until the
    next update; the slave's watchdog stops them if updates stop arriving.
    """
    global _servo_online
    vels = {name: float(w[i]) for i, name in enumerate(ALIGN_SERVOS)}
    try:
        r = _requests.post(f"{SERVO_PI_URL}/servo/velocity",
                           json={"velocities": vels}, timeout=1.0)
        _servo_online = r.ok
        return r.ok
    except Exception:
        _servo_online = False
        return False


def _get_align_dot_pos() -> Optional[np.ndarray]:
    """Current measured position of the dot selected for alignment, or None
    if it isn't being tracked this frame."""
    with _lock:
        st = dict(_tracking_state)
    for d in st.get("dots", []):
        # dot labels are "Dot A" / "Dot B"; _align_dot is the trailing letter
        if str(d.get("label", "")).strip().upper().endswith(_align_dot):
            if d.get("lost_frames", 0) > 0:
                return None
            return np.array([float(d["x"]), float(d["y"])])
    return None


def _lqr_gain(J: np.ndarray) -> np.ndarray:
    """LQR gain K (3×2) for the integrator x_{k+1} = x + J·u, so u = −K·x.

    A = I₂, B = J, cost Σ xᵀQx + uᵀRu.  Reuses the same closed-form as the
    module-level _LQR_K solve.  Falls back to a Tikhonov-regularized inverse if
    the DARE is unavailable or fails (e.g. J rank-deficient when the servos push
    along nearly the same image direction)."""
    Q = np.diag(ALIGN_Q)          # 2×2
    R = ALIGN_R * np.eye(3)       # 3×3
    if _HAVE_DARE:
        try:
            P = _dare_solve(np.eye(2), J, Q, R)
            return np.linalg.inv(R + J.T @ P @ J) @ (J.T @ P)   # A = I
        except Exception:
            pass
    # damped least-squares fallback: u = −(JᵀQJ + R)⁻¹ JᵀQ x
    JtQ = J.T @ Q
    return np.linalg.solve(JtQ @ J + R, JtQ)


def auto_align_loop() -> None:
    """Continuous image-based visual servoing: drive the selected dot to the
    target by streaming smooth servo *velocities* — the servos run continuously,
    never in discrete jogs.

    The dot's position is 2-D; the three servos form an over-actuated set whose
    effect is captured by a 2×3 velocity Jacobian G (column i = dot image
    velocity, px/s, per unit velocity command on servo i).

      1. Identify — any unmeasured column is bootstrapped by running that servo
         at ALIGN_PROBE_SPEED until the dot has travelled a measurable distance,
         then G[:,i] = (observed dot velocity) / (probe velocity).

      2. Correct — every ALIGN_LOOP_DT seconds, an LQR regulator on the
         sampled-integrator model x_{k+1} = x + (G·dt)·w (state x = pos − target,
         control w = servo velocities) gives w = −K·x = K·e, e = target − pos,
         K = _lqr_gain(G·dt).  The velocity command is streamed to the slave
         (POST /servo/velocity) and simply *updated* each period — the servos
         never stop, so motion is smooth and continuous.  Velocities are capped
         at ±ALIGN_MAX_SPEED; once within ALIGN_TOL_PX the servos are halted.

      3. Adapt — each period G is refined by an LMS step from the observed dot
         velocity:  G += μ·(v_obs − G·w)·wᵀ / (wᵀw).  Tracks drift / changing
         mirror response with no recalibration.

    Safety: whenever the dot can't be measured (lost) or auto-align is off, the
    servos are commanded to zero; the slave's watchdog is a second line of
    defence if updates ever stop arriving.
    """
    global _align_jac, _align_known

    LMS_GATE = 0.5   # px — minimum per-period travel to trust an LMS update
    _moving  = False

    def _halt():
        nonlocal _moving
        if _moving:
            _servo_velocity(np.zeros(3))
            _moving = False

    while True:
        if not _auto_align:
            _halt()
            sleep(0.1)
            continue

        pos = _get_align_dot_pos()
        if pos is None:                 # lost the dot — stop and wait
            _halt()
            sleep(ALIGN_LOOP_DT)
            continue

        # ── Phase 1: identify a servo's velocity effect by running it briefly ──
        if not all(_align_known):
            i      = _align_known.index(False)
            before = pos
            w      = np.zeros(3)
            w[i]   = ALIGN_PROBE_SPEED
            t0     = perf_counter()
            after  = pos
            while perf_counter() - t0 < ALIGN_PROBE_TMAX:
                if not _auto_align:
                    break
                _servo_velocity(w)       # refresh each period to feed the watchdog
                _moving = True
                sleep(ALIGN_LOOP_DT)
                meas = _get_align_dot_pos()
                if meas is None:         # lost the dot mid-probe — retry this servo
                    after = None
                    break
                after = meas
                if float(np.linalg.norm(after - before)) >= ALIGN_MIN_DISP:
                    break
            dt = perf_counter() - t0
            _halt()
            if after is None or dt <= 0.0:
                continue
            # Velocity Jacobian column: observed dot velocity per unit command.
            _align_jac[:, i] = (after - before) / dt / ALIGN_PROBE_SPEED
            _align_known[i]  = True
            continue

        err     = np.array(_target, dtype=float) - pos
        err_mag = float(np.linalg.norm(err))
        if err_mag <= ALIGN_TOL_PX:
            _halt()                      # arrived — hold position
            sleep(ALIGN_LOOP_DT)
            continue

        # ── Phase 2: continuous LQR velocity command ─────────────────────────
        K = _lqr_gain(_align_jac * ALIGN_LOOP_DT)   # B = G·dt (px per period per unit vel)
        w = K @ err                                 # w = K·e, servo velocities
        w = np.clip(w, -ALIGN_MAX_SPEED, ALIGN_MAX_SPEED)

        before = pos
        _servo_velocity(w)               # update velocities; servos keep running
        _moving = True
        sleep(ALIGN_LOOP_DT)

        # ── Phase 3: LMS refinement of G from the observed dot velocity ──────
        after = _get_align_dot_pos()
        if after is not None:
            disp = after - before
            ww   = float(w @ w)
            if float(np.linalg.norm(disp)) >= LMS_GATE and ww > 0:
                v_obs       = disp / ALIGN_LOOP_DT          # px/s
                residual    = v_obs - _align_jac @ w        # 2-vector
                _align_jac += ALIGN_LMS_RATE * np.outer(residual, w) / ww


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
    """Push annotated JPEG frames as binary WebSocket messages.

    Only sends when the capture loop has produced a *new* frame, so we never
    flood the socket with duplicate JPEGs (which congests the link and causes
    stutter on the Pi's Wi-Fi).
    """
    last_sent = -1
    while True:
        with _lock:
            data = _jpeg_frame
            idx  = _frame_idx
        if data and idx != last_sent:
            try:
                ws.send(data)
            except Exception:
                break
            last_sent = idx
        sleep(1 / (TARGET_FPS * 2))   # poll faster than frame rate; only send new frames


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
    data["auto_align"]   = _auto_align
    data["servo_online"] = _servo_online
    data["align"]        = _align_state()
    data["camera"] = {
        "gain":        round(float(_cam_gain), 2),
        "exposure_us": _cam_exposure_us,
        "exp_min":     CAM_EXP_MIN,
        "exp_max":     CAM_EXP_MAX,
        "thresh":      _thresh,
        "tuning":      _autotune_busy,
    }

    # ── LQR observer: compute state and control from current dot positions ────
    dots = data.get("dots", [])
    if len(dots) >= 2:
        mid_x = (dots[0]["x"]     + dots[1]["x"])     / 2.0
        mid_y = (dots[0]["y"]     + dots[1]["y"])     / 2.0
        de_x  = (dots[0]["vel_x"] + dots[1]["vel_x"]) / 2.0
        de_y  = (dots[0]["vel_y"] + dots[1]["vel_y"]) / 2.0
    else:
        mid_x = FRAME_WIDTH  / 2.0
        mid_y = FRAME_HEIGHT / 2.0
        de_x  = de_y = 0.0

    lqr_x = np.array([mid_x - FRAME_WIDTH  / 2.0,
                      mid_y - FRAME_HEIGHT / 2.0,
                      de_x, de_y])
    lqr_u = -_LQR_K @ lqr_x

    data["lqr"] = {
        "state":   [round(float(v), 3) for v in lqr_x],
        "control": [round(float(v), 3) for v in lqr_u],
        "jogs":    [int(round(float(v))) for v in lqr_u],
        "K":       [[round(float(v), 5) for v in row] for row in _LQR_K],
        "A":       [[round(float(v), 5) for v in row] for row in _LQR_A],
        "B":       [[round(float(v), 5) for v in row] for row in _LQR_B],
        "dt":      float(_LQR_dt),
    }
    return jsonify(data)


@app.route("/servo", methods=["POST"])
def servo_cmd():
    """Proxy a servo command to the slave Pi."""
    body = request.get_json(silent=True) or {}
    cmd  = str(body.get("command", "")).strip().lower()
    try:
        r = _requests.post(f"{SERVO_PI_URL}/servo",
                           json={"command": cmd}, timeout=1.5)
        global _servo_online
        _servo_online = r.ok
        return jsonify(r.json()), r.status_code
    except Exception as exc:
        _servo_online = False
        return jsonify({"error": str(exc)}), 503


@app.route("/servo/trim", methods=["GET", "POST"])
def servo_trim():
    """Proxy trim GET/POST to the servo Pi."""
    try:
        if request.method == "GET":
            r = _requests.get(f"{SERVO_PI_URL}/servo/trim", timeout=1.5)
        else:
            r = _requests.post(f"{SERVO_PI_URL}/servo/trim",
                               json=request.get_json(silent=True) or {},
                               timeout=1.5)
        global _servo_online
        _servo_online = r.ok
        return jsonify(r.json()), r.status_code
    except Exception as exc:
        _servo_online = False
        return jsonify({"error": str(exc)}), 503


@app.route("/servo/auto", methods=["POST"])
def servo_auto():
    """Enable or toggle the auto-alignment controller."""
    global _auto_align
    body        = request.get_json(silent=True) or {}
    _auto_align = bool(body.get("enabled", not _auto_align))
    return jsonify({"auto_align": _auto_align})


def _align_state() -> Dict[str, Any]:
    return {
        "enabled":  _auto_align,
        "dot":      _align_dot,
        "servos":   list(ALIGN_SERVOS),
        "target":   [round(float(_target[0]), 1), round(float(_target[1]), 1)],
        "identified": bool(all(_align_known)),
        "known":    list(_align_known),
        # Jacobian columns = per-servo image displacement per jog (for the UI)
        "jacobian": [[round(float(_align_jac[r, c]), 3) for c in range(3)]
                     for r in range(2)],
    }


def _reset_jacobian() -> None:
    """Forget the learned Jacobian so the loop re-identifies all servos."""
    global _align_jac, _align_known
    _align_jac   = np.zeros((2, 3))
    _align_known = [False, False, False]


@app.route("/align/config", methods=["GET", "POST"])
def align_config():
    """Configure the dot→target align controller (3 servos, one dot).

    POST JSON (any subset):
      enabled    bool    — run/stop the control loop
      dot        "A"/"B" — which tracked dot to drive
      target_x   float   — absolute target x (px)
      target_y   float   — absolute target y (px)
      target_dx  float   — nudge target x (px)
      target_dy  float   — nudge target y (px)
      reidentify bool    — discard the learned Jacobian and re-probe all servos

    Switching the controlled dot also forces re-identification, since each dot
    can respond differently to the servos.
    """
    global _auto_align, _align_dot
    if request.method == "GET":
        return jsonify(_align_state())

    body = request.get_json(silent=True) or {}

    if "enabled" in body:
        _auto_align = bool(body["enabled"])

    if "dot" in body:
        d = str(body["dot"]).upper()
        if d in ("A", "B") and d != _align_dot:
            _align_dot = d
            _reset_jacobian()

    if body.get("reidentify"):
        _reset_jacobian()

    if "target_x" in body:
        _target[0] = max(0.0, min(float(FRAME_WIDTH),  float(body["target_x"])))
    if "target_y" in body:
        _target[1] = max(0.0, min(float(FRAME_HEIGHT), float(body["target_y"])))
    if "target_dx" in body:
        _target[0] = max(0.0, min(float(FRAME_WIDTH),  _target[0] + float(body["target_dx"])))
    if "target_dy" in body:
        _target[1] = max(0.0, min(float(FRAME_HEIGHT), _target[1] + float(body["target_dy"])))

    return jsonify(_align_state())


@app.route("/control", methods=["POST"])
def control():
    global _needs_reset, _thresh, _overlay_on, _cam_exposure_req
    body = request.get_json(silent=True) or {}
    if body.get("action") == "reset":
        _needs_reset = True
    if "thresh" in body:
        _thresh = max(20, min(250, int(body["thresh"])))
    if "overlay" in body:
        _overlay_on = bool(body["overlay"])
    if "exposure_us" in body:
        # Serviced by the capture thread (it owns the camera).
        _cam_exposure_req = max(CAM_EXP_MIN, min(CAM_EXP_MAX, int(body["exposure_us"])))
    return jsonify({"ok": True})


@app.route("/camera/auto", methods=["POST"])
def camera_auto():
    """Request a camera auto-tune for best dot-acquisition conditions.

    The tune runs in the capture thread (which owns the camera): it servos the
    gain so the dots sit just below saturation, sets the detection threshold,
    and re-seeds the dots.  Returns immediately; poll /state ("camera".tuning)
    for completion and the resulting gain/threshold.
    """
    global _autotune_req
    if not _autotune_busy:
        _autotune_req = True
    return jsonify({"tuning": True, "busy": _autotune_busy})


# ─────────────────────────── Entry point ─────────────────────────────────────

if __name__ == "__main__":
    threading.Thread(target=capture_loop,   daemon=True).start()
    threading.Thread(target=auto_align_loop, daemon=True).start()
    print(f"HTTP server on port {WEB_PORT}")
    print(f"Servo Pi URL: {SERVO_PI_URL}")
    app.run(host="0.0.0.0", port=WEB_PORT, threaded=True)
