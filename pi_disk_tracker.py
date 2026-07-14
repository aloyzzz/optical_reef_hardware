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
import os

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
BLOB_MIN_CIRC    = 0.65
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

# ── Segments: one steerable mirror (servo Pi) per tracked dot ────────────────
# Each dot (label "A"/"B") is driven by its own mirror on its own servo Pi.  URLs
# are env-overridable so a DHCP address change never needs a code edit — set
# SERVO_PI_URL / SERVO_PI_URL_1 in the environment (or ~/.servo_last_ip via the
# launcher) instead of hardcoding.  Add more dicts here for >2 segments.
#
# `dot`     — which tracked dot this mirror steers (its beam's spot on the sensor)
# `servos`  — the two motorised screws used for tip/tilt, in Jacobian-column order
# `enabled` — start driving this segment as soon as auto-align is on
SEGMENTS_CONFIG = [
    {"name": "s0",                      # dot 0
     "url":     os.environ.get("SERVO_PI_URL",   "http://10.63.149.96:5000"),
     "dot":     "A",                    # first tracked dot  (labelled "Dot A")
     "servos":  ("A", "B"),             # tip/tilt screws used for alignment (A/B/C on the mount)
     "enabled": True},
    {"name": "s1",                      # dot 1
     "url":     os.environ.get("SERVO_PI_URL_1", "http://10.63.149.93:5000"),
     "dot":     "B",                    # second tracked dot (labelled "Dot B")
     "servos":  ("A", "B"),
     "enabled": True},                  # s1 is wired up — drive it whenever auto-align is on
]

# Backward-compatible alias — legacy manual-jog routes default to the first segment.
SERVO_PI_URL          = SEGMENTS_CONFIG[0]["url"]
AUTO_ALIGN_SCORE_TARGET = 0.92   # alignment_score above which auto-align considers done
AUTO_ALIGN_SETTLE     = 0.35     # seconds between jog and next measurement

# ── "Drive dot to target" controller ─────────────────────────────────────────
# Probes each servo (regression over timestamped frames → accurate Jacobian
# despite network lag), then runs a continuous LQR velocity controller:
#   state  x = [ex, ey, vx, vy]   (position error + dot velocity, px and px/s)
#   input  w = servo velocities    (N-vector, units relative to STOP)
#   w = −K·x   where K = lqr(A, B, Q, R)
# Dot velocity is derived from consecutive frame timestamps (not the Kalman
# filter), so the dt used for both the LQR model and the LMS Jacobian update
# matches the actual camera frame interval rather than the nominal loop period.
ALIGN_SERVOS      = ("A", "B") # servos driving the dot, in Jacobian-column order
ALIGN_TOL_PX      = 1.5        # stop correcting once error is within this many px
ALIGN_MAX_SPEED   = 0.40       # hard cap on per-servo velocity command
ALIGN_MAX_DOT_SPEED = 45.0     # px/s — hard cap on how fast the dot itself is driven
ALIGN_PROBE_SPEED = 0.10       # velocity used during Jacobian identification (gentle:
                               # a slow probe keeps the calibration step small)
ALIGN_PROBE_TMAX  = 3.0        # max seconds per probe (longer, since the probe is slower)
ALIGN_PROBE_MIN_T = 0.30       # minimum probe duration before early-exit is allowed
ALIGN_MIN_DISP    = 1.0        # px — minimum dot displacement to trust a probe result
ALIGN_PROBE_MAX_DISP = 20.0    # px — abort the probe once the dot has moved this far
ALIGN_LOOP_DT     = 0.04       # control loop update period (seconds)
ALIGN_SETTLE_DT   = 0.20       # post-probe settle: wait for dot to stop after halting

# ── Probe-based dot discovery ────────────────────────────────────────────────
# A segment does NOT trust the `dot` field in SEGMENTS_CONFIG.  Probing is
# serialised across segments by _IDENTIFY_LOCK, so during a probe exactly one
# mirror is moving — whichever dot moves *is* that mirror's dot, by definition.
# The probe therefore watches every tracked dot and binds the segment to the one
# that moved, provided the motion is unambiguous:
#   • the winner moved at least ALIGN_MIN_DISP px, and
#   • it moved at least ALIGN_BIND_DOMINANCE × as far as any other dot.
# The config's `dot` is kept only as a hint (for display before binding).
ALIGN_BIND_DOMINANCE = 3.0

# ── Symmetric wiggle probe ───────────────────────────────────────────────────
# Identification drives the servo one way until the dot has moved
# ALIGN_WIGGLE_DISP px, then drives it back the other way for the same duration.
# Net displacement is ~0, so a probe is safe even when the dot starts near a
# frame border, and the out-and-back difference cancels any slow drift.
ALIGN_WIGGLE_DISP = 6.0        # px — target displacement for each half of the wiggle

# ── Runaway guard ────────────────────────────────────────────────────────────
# A wrong-sign or noise-fitted Jacobian drives the dot away from the target at
# full speed.  When the dot's observed velocity opposes the commanded velocity
# (or the error simply keeps growing) for this many consecutive control periods,
# the segment halts, throws away its Jacobian and dot binding, and re-identifies.
ALIGN_RUNAWAY_N   = 5          # consecutive bad periods before forcing re-identification
ALIGN_RUNAWAY_V   = 2.0        # px/s — ignore periods where the dot barely moved

# ── Frame-border keep-out ────────────────────────────────────────────────────
# The dot is never driven outside a safe box inset BORDER_FRAC of the frame on
# each side.  Within BORDER_SOFT_PX of that box's edge the outward component of
# the commanded dot velocity is faded to zero; past the edge the controller
# ignores the target and pushes the dot back inward at BORDER_PUSH_SPEED.
BORDER_FRAC       = 0.10       # 10 % of frame width/height reserved on each side
BORDER_SOFT_PX    = 20.0       # px — slow-down band just inside the safe box
BORDER_PUSH_SPEED = 25.0       # px/s — inward recovery speed when outside the box
# LQR weights
ALIGN_Q           = (1.5, 1.5) # position-error penalty (x, y)
ALIGN_R           = 0.12       # control-effort penalty; larger → gentler/slower
ALIGN_BETA        = 10.0       # velocity decay rate (s⁻¹) in the LQR plant model
ALIGN_LMS_RATE    = 0.06       # Jacobian online adaptation rate

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
_intersect_sr    = float(INTERSECTION_SEARCH_RADIUS)   # set here and in calibrate_tracking_params
_ring_edges_d    = [0, 6, 12, 20, 30]
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
KF_R = np.diag([1.0, 1.0])              # measurement noise: ~±1 px (aperture centroid on Airy disk)
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
COL_BORDER = (70, 70, 70)     # grey — keep-out border box

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
_auto_align     = False   # master switch for all segment align controllers

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
# Live image controls (applied by the capture thread, which owns picam2). Initial
# values mirror the fixed dot-detection setup applied at startup.
_cam_contrast   = 2.0     # 1.0 = normal, 2.0 = dot-popping
_cam_brightness = 0.0     # -1..1
_cam_ae         = False   # auto-exposure enabled? (True gives a normal image)
_cam_ctrl_req   = {}      # pending picam2 controls from the UI (dict)

# ── Align controllers: one Segment per steerable mirror / tracked dot ────────
# `_auto_align` (declared above) is the master switch; each Segment additionally
# has its own `enabled` flag, Jacobian, target and control-loop runtime state so
# multiple mirrors are driven independently and concurrently (one thread each).
# Probing (Jacobian identification) is serialised across segments by
# `_IDENTIFY_LOCK` so two mirrors never move at once during identification —
# that keeps each probe's dot-displacement measurement unambiguous.
_IDENTIFY_LOCK = threading.Lock()

# The segment currently probing, or None.  Probing infers "the dot that moved is
# my dot", which is only sound while every *other* mirror is still — so segments
# already in closed-loop control halt themselves whenever this is set to someone
# else.  _IDENTIFY_LOCK alone isn't enough: it serialises probes against probes,
# not probes against control.
_PROBER: Optional["Segment"] = None


class Segment:
    """A mirror + servo Pi that steers one tracked dot to a pixel target."""

    def __init__(self, name, url, dot, servos, enabled=True):
        self.name    = str(name)
        self.url     = str(url)
        self.dot_hint = str(dot).upper()      # config's guess at this mirror's dot
        self.dot     = self.dot_hint          # working label; replaced by the probe
        self.bound   = False                  # True once a probe proved which dot this is
        self.servos  = tuple(servos)          # screw names, Jacobian-column order
        self.enabled = bool(enabled)
        self.online  = False
        self.n       = len(self.servos)
        self.jac     = np.zeros((2, self.n))  # image Jacobian: col i = px/s per unit vel of servo i
        self.known   = [False] * self.n       # whether each column has been identified
        self.probe_sign = [1.0] * self.n      # probe direction per servo; flipped when a
                                              # probe pushes the dot out of the safe box
        self.target  = [FRAME_WIDTH / 2.0, FRAME_HEIGHT / 2.0]
        # control-loop runtime state (owned by this segment's align thread)
        self.ts_prev = None                   # (timestamp, pos) for velocity estimate
        self.moving  = False
        self.bad     = 0                      # consecutive runaway-guard violations
        self.fault   = ""                     # last fault reason, surfaced in /state
        self.needs_rebind = False             # set by the capture thread after a dot reset

    # ── networking to this segment's servo Pi ──
    def post(self, cmd: str) -> bool:
        """Send one discrete command (e.g. 'a+', 'tip-') to this segment."""
        for _attempt in range(2):
            try:
                r = _requests.post(f"{self.url}/servo",
                                   json={"command": cmd}, timeout=2.0)
                self.online = r.ok
                return r.ok
            except Exception:
                if _attempt == 0:
                    sleep(0.06)
        self.online = False
        return False

    def servo_velocity(self, w) -> bool:
        """Stream continuous per-servo velocities to this segment (non-blocking)."""
        vels = {name: float(w[i]) for i, name in enumerate(self.servos)}
        try:
            r = _requests.post(f"{self.url}/servo/velocity",
                               json={"velocities": vels}, timeout=1.0)
            self.online = r.ok
            return r.ok
        except Exception:
            self.online = False
            return False

    def halt(self) -> None:
        if self.moving:
            self.servo_velocity(np.zeros(self.n))
            self.moving = False

    # ── this segment's dot in the shared tracking state ──
    def get_dot_timestamped(self) -> Optional[Tuple[float, np.ndarray]]:
        """Frame timestamp and position of this segment's dot, or None if lost."""
        with _lock:
            st = dict(_tracking_state)
        t = float(st.get("timestamp", perf_counter()))
        for d in st.get("dots", []):
            if str(d.get("label", "")).strip().upper().endswith(self.dot):
                if d.get("lost_frames", 0) > 0:
                    return None
                return t, np.array([float(d["x"]), float(d["y"])])
        return None

    def get_dot_pos(self) -> Optional[np.ndarray]:
        r = self.get_dot_timestamped()
        return r[1] if r is not None else None

    def reset_jacobian(self) -> None:
        self.jac   = np.zeros((2, self.n))
        self.known = [False] * self.n

    def unbind(self) -> None:
        """Forget which dot this mirror drives; the next probe re-discovers it."""
        self.bound = False
        self.dot   = self.dot_hint
        self.reset_jacobian()
        self.ts_prev = None
        self.bad     = 0

    def bind_dot(self, label: str) -> bool:
        """Claim `label` as this mirror's dot.  False if another segment owns it."""
        for other in SEGMENTS:
            if other is not self and other.bound and other.dot == label:
                self.fault = (f"dot {label} already bound to segment {other.name} — "
                              f"two mirrors appear to drive the same dot")
                print(f"[align:{self.name}] {self.fault}")
                return False
        self.dot   = label
        self.bound = True
        self.fault = ""
        print(f"[align:{self.name}] bound to Dot {label} "
              f"(config hint was {self.dot_hint})")
        return True

    def state_dict(self) -> Dict[str, Any]:
        tgt = self.target if (self.target and len(self.target) >= 2) \
            else [FRAME_WIDTH / 2, FRAME_HEIGHT / 2]
        return {
            "name":       self.name,
            "url":        self.url,
            "dot":        self.dot,
            "servos":     list(self.servos),
            "enabled":    self.enabled,
            "online":     self.online,
            "bound":      self.bound,
            "dot_hint":   self.dot_hint,
            "fault":      self.fault,
            "target":     [round(float(tgt[0]), 1), round(float(tgt[1]), 1)],
            "identified": bool(all(self.known)),
            "known":      list(self.known),
            "jacobian":   [[round(float(self.jac[r, c]), 3) for c in range(self.n)]
                           for r in range(2)],
        }


SEGMENTS: List[Segment] = [
    Segment(c["name"], c["url"], c["dot"], c["servos"], c.get("enabled", True))
    for c in SEGMENTS_CONFIG
]
SEGMENTS_BY_NAME: Dict[str, Segment] = {s.name: s for s in SEGMENTS}


def _segment_for(body: Dict[str, Any]) -> Segment:
    """Pick the segment a request targets: body['segment'] name, else the first."""
    name = str(body.get("segment", "")).strip()
    return SEGMENTS_BY_NAME.get(name, SEGMENTS[0])

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
    """Find n brightest isolated spots in a single pass.

    Thresholds the top-hat image at a robust percentile of the peak brightness,
    then returns the top-n components by brightness. Simple, direct, no loops.
    """
    _, blurred = get_proc_blurred(gray)
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
    """Derive tracking constants from the initial dots' PSF.

    Measures FWHM of both dots and scales all radii/distances to match the optics.
    Also initializes Kalman velocity from the current separation vector.
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
            d.psf_ref = {k: psf[k] for k in ('peak', 'flux', 'fwhm_mean')}

    if not fwhms:
        print("[calibrate] PSF measurement failed — keeping defaults")
        return

    avg_fwhm = float(np.mean(fwhms))

    # Aperture: ~2.5× FWHM to capture the disk core plus noise (Airy disk total)
    ap_r = max(avg_fwhm * 2.5, 10.0)

    _fwhm_calib      = avg_fwhm
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

    # Keep-out border: the controller never drives a dot outside this box.
    bx0, by0, bx1, by1 = safe_box()
    cv2.rectangle(out, (int(bx0), int(by0)), (int(bx1), int(by1)),
                  COL_BORDER, 1, cv2.LINE_AA)

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
    """Adjust AnalogueGain so the dot cores sit comfortably below saturation.

    Measures brightness within the tracked dot apertures rather than the global
    frame max, so background reflections or bright edges don't fool the servo.
    Falls back to the global frame max only when no dot positions are available.

    Does NOT force a dot reset — the Kalman filter keeps tracking through the
    gain change. A reset is only triggered if the dots were already lost.
    """
    global _cam_gain, _thresh, _needs_reset

    def _aperture_peak(g: np.ndarray) -> float:
        """Peak brightness within the tracked dot apertures, or global max."""
        with _lock:
            st = dict(_tracking_state)
        dot_entries = st.get("dots", [])
        peaks = []
        for d in dot_entries:
            if d.get("lost_frames", 1) == 0:
                r = int(np.ceil(_ap_r))
                x, y = int(round(d["x"])), int(round(d["y"]))
                x0 = max(0, x - r);  x1 = min(g.shape[1], x + r + 1)
                y0 = max(0, y - r);  y1 = min(g.shape[0], y + r + 1)
                roi = cv2.medianBlur(g, 3)[y0:y1, x0:x1]
                if roi.size:
                    peaks.append(float(roi.max()))
        if peaks:
            return max(peaks)
        # Fallback: global max (no live dot positions)
        return float(cv2.medianBlur(g, 3).max())

    # CAM_TARGET_PEAK is 235, but we aim 20 DN lower (215) to give headroom
    # against a one-step gain overshoot clipping the PSF.
    target = CAM_TARGET_PEAK - 20.0
    tol    = CAM_PEAK_TOL

    gain = float(_cam_gain)
    gray = _grab_gray(picam2)
    for _ in range(10):
        peak = _aperture_peak(gray)
        if abs(peak - target) <= tol:
            break
        if peak >= 253.0:
            ratio = 0.5
        else:
            ratio = float(np.clip(target / max(peak, 1.0), 0.5, 2.0))
        new_gain = float(np.clip(gain * ratio, CAM_GAIN_MIN, CAM_GAIN_MAX))
        if abs(new_gain - gain) < 1e-3:
            break
        gain = new_gain
        picam2.set_controls({"AnalogueGain": gain})
        sleep(0.25)
        gray = _grab_gray(picam2)

    _cam_gain = gain

    # Set detection threshold from the preprocessed (top-hat) image at the dot
    # apertures — the same image space that detect_blobs uses.
    proc    = preprocess_frame(gray)
    blurred = cv2.GaussianBlur(proc, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
    with _lock:
        st = dict(_tracking_state)
    dot_entries  = st.get("dots", [])
    proc_peaks   = []
    dots_tracked = True
    for d in dot_entries:
        if d.get("lost_frames", 1) == 0:
            r = int(np.ceil(_ap_r))
            x, y = int(round(d["x"])), int(round(d["y"]))
            x0 = max(0, x - r);  x1 = min(blurred.shape[1], x + r + 1)
            y0 = max(0, y - r);  y1 = min(blurred.shape[0], y + r + 1)
            roi = blurred[y0:y1, x0:x1]
            if roi.size:
                proc_peaks.append(float(roi.max()))
        else:
            dots_tracked = False

    proc_peak = max(proc_peaks) if proc_peaks else float(blurred.max())
    _thresh   = int(np.clip(proc_peak * 0.40, 5, 100))

    # Only reset if dots were already lost — no need to discard a live track.
    if not dots_tracked:
        _needs_reset = True

    print(f"[camera] auto-tune → gain={_cam_gain:.2f}  aperture_peak={_aperture_peak(gray):.0f}  "
          f"proc_peak={proc_peak:.0f}  thresh={_thresh}  reset={not dots_tracked}")


def capture_loop():
    global _jpeg_frame, _tracking_state, _thresh, _frame_idx
    global _fps_actual, _intersecting, _needs_reset
    global _cam_exposure_us, _cam_exposure_req, _autotune_req, _autotune_busy
    global _cam_ctrl_req

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

        # Apply pending image controls from the UI (contrast/brightness/gain/AE/reset).
        if _cam_ctrl_req:
            req, _cam_ctrl_req = dict(_cam_ctrl_req), {}
            try:
                picam2.set_controls(req)
            except Exception as _c_e:
                print(f"[camera] control set failed ({req}): {_c_e}")

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
            # Re-seeding sorts the dots left-to-right, so "Dot A" may now be the
            # other physical dot.  Every mirror's binding is stale — make each
            # align thread re-probe rather than steer a dot it no longer owns.
            for _s in SEGMENTS:
                _s.needs_rebind = True
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
            "timestamp":     t_now,          # perf_counter() time of this frame
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

# Servo I/O and dot lookup now live on Segment (see the Segment class above):
#   Segment.post / .servo_velocity / .halt      — command this segment's servo Pi
#   Segment.get_dot_timestamped / .get_dot_pos  — locate this segment's dot


def _dots_snapshot() -> Tuple[float, Dict[str, np.ndarray]]:
    """(frame timestamp, {dot label → position}) for every currently-tracked dot.

    Dots the tracker has lost are omitted.  The probe uses this to see which dot
    actually moved, rather than assuming it already knows.
    """
    with _lock:
        st = dict(_tracking_state)
    t    = float(st.get("timestamp", perf_counter()))
    dots: Dict[str, np.ndarray] = {}
    for d in st.get("dots", []):
        if d.get("lost_frames", 0) > 0:
            continue
        label = str(d.get("label", "")).strip().upper()[-1:]
        if label:
            dots[label] = np.array([float(d["x"]), float(d["y"])])
    return t, dots


def _moved_dot(first: Dict[str, np.ndarray],
               last:  Dict[str, np.ndarray],
               strict: bool = True) -> Tuple[Optional[str], float]:
    """Which dot moved during a probe, and by how far.

    Returns (label, displacement) for the dot that moved furthest.  Under
    `strict` (the default, used for binding) the result must be unambiguous: the
    winner must have travelled at least ALIGN_MIN_DISP px and at least
    ALIGN_BIND_DOMINANCE × as far as every other dot, otherwise (None, …) is
    returned — better to re-probe than to bind a mirror to the wrong dot.
    Non-strict just names the furthest-moving dot; used for border recovery,
    where a best guess beats no guess.
    """
    disps = {lbl: float(np.linalg.norm(last[lbl] - first[lbl]))
             for lbl in first if lbl in last}
    if not disps:
        return None, 0.0
    ranked = sorted(disps.items(), key=lambda kv: kv[1], reverse=True)
    best, best_d = ranked[0]
    if not strict:
        return best, best_d
    if best_d < ALIGN_MIN_DISP:
        return None, best_d
    if len(ranked) > 1 and best_d < ALIGN_BIND_DOMINANCE * max(ranked[1][1], 1e-6):
        return None, best_d
    return best, best_d


def safe_box() -> Tuple[float, float, float, float]:
    """(x_min, y_min, x_max, y_max) of the region the dot is allowed to reach."""
    mx = BORDER_FRAC * FRAME_WIDTH
    my = BORDER_FRAC * FRAME_HEIGHT
    return mx, my, FRAME_WIDTH - mx, FRAME_HEIGHT - my


def clamp_to_safe(x: float, y: float) -> Tuple[float, float]:
    x0, y0, x1, y1 = safe_box()
    return float(np.clip(x, x0, x1)), float(np.clip(y, y0, y1))


def _outside_safe(pos: np.ndarray, inset: float = 0.0) -> bool:
    """True when `pos` is outside the safe box, optionally shrunk by `inset` px."""
    x0, y0, x1, y1 = safe_box()
    return not (x0 + inset <= float(pos[0]) <= x1 - inset
                and y0 + inset <= float(pos[1]) <= y1 - inset)


def _border_guard(pos: np.ndarray,
                  v_des: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Constrain a desired dot velocity so the dot stays inside the safe box.

    Per axis: past the box edge the velocity is replaced by an inward push;
    inside the BORDER_SOFT_PX band the *outward* component is faded linearly to
    zero at the edge, so the dot decelerates into the border instead of hitting
    it. Motion inward is never restricted.

    Returns (velocity, outside) where `outside` is True if the dot has already
    left the safe box on either axis.
    """
    x0, y0, x1, y1 = safe_box()
    lo = np.array([x0, y0])
    hi = np.array([x1, y1])
    v  = np.array(v_des, dtype=float)
    outside = False
    for i in (0, 1):
        p = float(pos[i])
        if p < lo[i]:
            v[i] = BORDER_PUSH_SPEED
            outside = True
        elif p > hi[i]:
            v[i] = -BORDER_PUSH_SPEED
            outside = True
        elif v[i] < 0 and (p - lo[i]) < BORDER_SOFT_PX:
            v[i] *= max(0.0, (p - lo[i]) / BORDER_SOFT_PX)
        elif v[i] > 0 and (hi[i] - p) < BORDER_SOFT_PX:
            v[i] *= max(0.0, (hi[i] - p) / BORDER_SOFT_PX)
    return v, outside


def _cap_dot_speed(v: np.ndarray) -> np.ndarray:
    """Scale a desired dot velocity down to ALIGN_MAX_DOT_SPEED, keeping direction."""
    s = float(np.linalg.norm(v))
    if s > ALIGN_MAX_DOT_SPEED > 0.0:
        return v * (ALIGN_MAX_DOT_SPEED / s)
    return v


def _servo_cmd_for(J: np.ndarray, v_des: np.ndarray) -> np.ndarray:
    """Servo velocities that produce dot velocity `v_des`, via damped least squares.

    Damping keeps the command finite when the Jacobian is near-singular (both
    servos moving the dot along nearly the same image direction).
    """
    lam = 1e-3 * max(1.0, float(np.linalg.norm(J)) ** 2)
    try:
        w = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(2), v_des)
    except np.linalg.LinAlgError:
        return np.zeros(J.shape[1])
    return np.clip(w, -ALIGN_MAX_SPEED, ALIGN_MAX_SPEED)


def _probe_recover(seg: "Segment", i: int, label: Optional[str]) -> None:
    """Reverse servo `i` until the dot it moves is safely back inside the safe box.

    Called when a probe pushes its dot into the border band. The Jacobian isn't
    known yet at this point, so we can't solve for an inward velocity — we just
    run the same servo backwards (seg.probe_sign[i] has already been flipped),
    which is the one motion we know undoes what just happened.  `label` is the dot
    the probe observed moving, which may not be the one the config claims.
    """
    w    = np.zeros(seg.n)
    w[i] = ALIGN_PROBE_SPEED * float(seg.probe_sign[i])
    t0   = perf_counter()
    while perf_counter() - t0 < ALIGN_PROBE_TMAX * 2:
        if not (_auto_align and seg.enabled):
            break
        seg.servo_velocity(w)
        seg.moving = True
        sleep(ALIGN_LOOP_DT)
        _, dots = _dots_snapshot()
        p = dots.get(label) if label else None
        if p is None or not _outside_safe(p, inset=BORDER_SOFT_PX):
            break
    seg.halt()
    sleep(ALIGN_SETTLE_DT)
    print(f"[align:{seg.name}] probe on servo {seg.servos[i]} hit the frame border — "
          f"backed off, will probe the other way")


def _probe_half(seg: "Segment",
                w: np.ndarray,
                tmax: float,
                stop_disp: Optional[float]) -> Tuple[List[Tuple[float, Dict[str, np.ndarray]]], str]:
    """Drive one servo velocity and sample *every* dot until a stop condition.

    Returns (samples, status) where each sample is (frame timestamp, {label: pos})
    and status is one of:
      'ok'      — reached stop_disp / tmax cleanly
      'border'  — the moving dot was heading out of the safe box; probe abandoned
      'lost'    — the tracker lost the dots
      'abort'   — auto-align was switched off mid-probe

    The border check fires while the dot is still inside the box (BORDER_SOFT_PX
    of margin) and only when it is moving *outward*, so a probe that starts near
    an edge but pushes inward isn't falsely aborted.
    """
    t0 = perf_counter()
    _, first = _dots_snapshot()
    if not first:
        return [], 'lost'
    centre  = np.array([FRAME_WIDTH / 2.0, FRAME_HEIGHT / 2.0])
    samples: List[Tuple[float, Dict[str, np.ndarray]]] = []

    while perf_counter() - t0 < tmax:
        if not (_auto_align and seg.enabled):
            return samples, 'abort'
        seg.servo_velocity(w)
        seg.moving = True
        sleep(ALIGN_LOOP_DT)

        t, dots = _dots_snapshot()
        common  = [lbl for lbl in first if lbl in dots]
        if not common:
            return samples, 'lost'
        samples.append((t, dots))

        mover = max(common, key=lambda l: float(np.linalg.norm(dots[l] - first[l])))
        pos   = dots[mover]
        moved = pos - first[mover]
        disp  = float(np.linalg.norm(moved))

        outward = float(np.dot(moved, centre - first[mover])) < 0.0
        if outward and _outside_safe(pos, inset=BORDER_SOFT_PX):
            return samples, 'border'
        if disp >= ALIGN_PROBE_MAX_DISP:
            return samples, 'ok'
        if (stop_disp is not None and disp >= stop_disp
                and perf_counter() - t0 >= ALIGN_PROBE_MIN_T):
            return samples, 'ok'

    return samples, 'ok'


def _fit_velocity(samples: List[Tuple[float, Dict[str, np.ndarray]]],
                  label: str) -> Optional[np.ndarray]:
    """Least-squares dot velocity (px/s) for `label` over a probe's samples.

    Regression over frame timestamps rather than endpoint differencing: servo
    start/stop lag distorts the endpoints but not the sustained linear motion in
    between, which the slope picks up.
    """
    pts = [(t, d[label]) for t, d in samples if label in d]
    if len(pts) < 3:
        return None
    ts = np.array([p[0] for p in pts])
    ts = ts - ts[0]
    if ts[-1] <= 0.01:
        return None
    A  = np.column_stack([ts, np.ones(len(ts))])
    vx = float(np.linalg.lstsq(A, np.array([p[1][0] for p in pts]), rcond=None)[0][0])
    vy = float(np.linalg.lstsq(A, np.array([p[1][1] for p in pts]), rcond=None)[0][0])
    return np.array([vx, vy])


def _identify_servo(seg: "Segment", i: int) -> bool:
    """Symmetric wiggle probe of servo `i`: discovers this mirror's dot and J[:, i].

    Drives the servo one way until its dot has moved ALIGN_WIGGLE_DISP px, then
    back the other way for the same wall-clock duration.  Net displacement is
    ~zero, so the probe is safe even when the dot starts near a frame border, and
    differencing the two half-velocities cancels any slow drift the dot had
    independently of the servo.

    Must be called with _IDENTIFY_LOCK held — the dot-discovery step is only valid
    while this is the only mirror moving.  Returns True when J[:, i] was set.
    """
    sign  = float(seg.probe_sign[i])
    w_fwd = np.zeros(seg.n)
    w_fwd[i] = ALIGN_PROBE_SPEED * sign

    t_start = perf_counter()
    fwd, status = _probe_half(seg, w_fwd, ALIGN_PROBE_TMAX, ALIGN_WIGGLE_DISP)
    dur_fwd = perf_counter() - t_start
    seg.halt()
    sleep(ALIGN_SETTLE_DT)

    if status == 'border':
        seg.probe_sign[i] = -sign
        mover, _ = _moved_dot(fwd[0][1], fwd[-1][1], strict=False) if len(fwd) >= 2 else (None, 0.0)
        _probe_recover(seg, i, mover)
        return False
    if status != 'ok' or len(fwd) < 3:
        return False

    # ── Who moved?  Whichever dot did is this mirror's dot, by definition. ──
    label, disp = _moved_dot(fwd[0][1], fwd[-1][1])
    if label is None:
        # Either nothing moved enough, or two dots moved comparably (another
        # mirror is drifting, or the tracker swapped identities).  Re-probe.
        return False

    if seg.bound:
        if label != seg.dot:
            seg.fault = (f"servo {seg.servos[i]} moves Dot {label} but this mirror is "
                         f"bound to Dot {seg.dot} — check servo wiring")
            print(f"[align:{seg.name}] {seg.fault}")
            return False
    elif not seg.bind_dot(label):
        return False

    # ── Wiggle back: same servo, opposite sign, same duration. ──
    v_fwd = _fit_velocity(fwd, label)
    if v_fwd is None:
        return False

    rev, rstatus = _probe_half(seg, -w_fwd, max(dur_fwd, ALIGN_PROBE_MIN_T), None)
    seg.halt()
    sleep(ALIGN_SETTLE_DT)

    v_rev = _fit_velocity(rev, label) if rstatus in ('ok', 'border') else None

    # J column = px/s per unit servo velocity.  With both halves we average the
    # two independent estimates (drift cancels); with only the forward half we
    # fall back to it alone.
    cmd = ALIGN_PROBE_SPEED * sign
    if v_rev is not None:
        col = 0.5 * (v_fwd - v_rev) / cmd
    else:
        col = v_fwd / cmd

    seg.jac[:, i] = col
    seg.known[i]  = True
    seg.fault     = ""
    print(f"[align:{seg.name}] servo {seg.servos[i]} → Dot {label} "
          f"({len(fwd)}+{len(rev)} frames, {disp:.1f}px out-and-back): "
          f"J[:,{i}] = {col.round(2)}")
    return True


def _lqr_gain(J: np.ndarray) -> np.ndarray:
    """LQR gain K (n×4) for state x = [ex, ey, vx, vy].

    Uses actual frame-derived dt in the plant model so the gain matches reality
    even when the control loop runs slower than ALIGN_LOOP_DT due to network lag.
    Falls back to a Tikhonov pseudo-inverse if DARE is unavailable.
    """
    dt  = ALIGN_LOOP_DT
    a22 = max(0.0, 1.0 - ALIGN_BETA * dt)
    # err = target - pos, so positive dot velocity reduces error: err(k+1) = err(k) - dt*vel
    A = np.array([[1.0, 0.0, -dt,  0.0],
                  [0.0, 1.0, 0.0,  -dt],
                  [0.0, 0.0, a22,  0.0],
                  [0.0, 0.0, 0.0,  a22]])
    n = J.shape[1]
    B = np.zeros((4, n))
    B[2, :] = J[0, :] * dt
    B[3, :] = J[1, :] * dt
    Q = np.diag([ALIGN_Q[0], ALIGN_Q[1], 0.5, 0.5])
    R = ALIGN_R * np.eye(n)
    if _HAVE_DARE:
        try:
            P = _dare_solve(A, B, Q, R)
            return np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        except Exception:
            pass
    J2  = J * dt
    JtQ = J2.T @ np.diag(ALIGN_Q)
    K2  = np.linalg.solve(JtQ @ J2 + R, JtQ)
    return np.hstack([K2, np.zeros((n, 2))])


def segment_align_loop(seg: "Segment") -> None:
    """Continuous LQR visual servoing for ONE segment (mirror → its dot).

    Runs in its own thread; multiple segments run concurrently and independently
    since each drives a different mirror and watches a different dot.  Jacobian
    identification (Phase 1) is serialised across segments via _IDENTIFY_LOCK so
    only one mirror probes at a time — that keeps every probe's dot-displacement
    measurement (and the tracker's A/B identity assignment) unambiguous.

      1. Identify — wiggle each servo out and back; the dot that moves is this
         mirror's dot (binding), and the regression slope gives J[:, i].
      2. Control — every ALIGN_LOOP_DT seconds compute dot velocity from
         consecutive frame timestamps, build x = [ex, ey, vx, vy], apply w = −K·x.
      3. Adapt — runaway guard, then an LMS step using the actual inter-frame dt.
    """
    global _PROBER

    LMS_GATE = 0.5      # px — minimum per-period displacement to trust LMS

    while True:
        if not (_auto_align and seg.enabled):
            seg.halt()
            seg.ts_prev = None
            sleep(0.1)
            continue

        if seg.needs_rebind:
            seg.needs_rebind = False
            seg.halt()
            seg.unbind()
            print(f"[align:{seg.name}] dots were re-seeded — re-identifying")

        ts_now = seg.get_dot_timestamped()
        if ts_now is None:              # lost the dot — stop and wait
            seg.halt()
            seg.ts_prev = None
            sleep(ALIGN_LOOP_DT)
            continue
        t_now, pos = ts_now

        # ── Phase 1: identify a servo, and discover which dot it drives ───────
        # A symmetric wiggle probe (out and back) leaves the dot where it started,
        # so this is safe even when the dot begins near a frame border.  Held
        # under _IDENTIFY_LOCK so no other mirror moves while we probe — that is
        # what makes "the dot that moved is my dot" a valid inference.
        if not all(seg.known):
            with _IDENTIFY_LOCK:
                if not (_auto_align and seg.enabled):
                    continue
                _PROBER = seg          # freeze the other mirrors while we probe
                try:
                    _identify_servo(seg, seg.known.index(False))
                finally:
                    _PROBER = None
            sleep(ALIGN_LOOP_DT)
            continue

        # Another mirror is probing: hold still, or its "which dot moved?" test
        # will see our dot move and bind to it.
        if _PROBER is not None and _PROBER is not seg:
            seg.halt()
            seg.ts_prev = None
            sleep(ALIGN_LOOP_DT)
            continue

        # ── Border keep-out: outside the safe box the target is ignored and the
        # dot is driven straight back inside before anything else happens.
        v_push, outside = _border_guard(pos, np.zeros(2))
        if outside:
            seg.servo_velocity(_servo_cmd_for(seg.jac, _cap_dot_speed(v_push)))
            seg.moving  = True
            seg.ts_prev = None
            sleep(ALIGN_LOOP_DT)
            continue

        # Target must be set before control starts
        if seg.target is None or any(not np.isfinite(x) for x in seg.target):
            seg.halt()
            sleep(ALIGN_LOOP_DT)
            continue

        err     = np.array(seg.target, dtype=float) - pos
        err_mag = float(np.linalg.norm(err))
        if err_mag <= ALIGN_TOL_PX:
            seg.halt()
            seg.ts_prev = None
            seg.bad     = 0
            sleep(ALIGN_LOOP_DT)
            continue

        # ── Phase 2: LQR velocity command ────────────────────────────────────
        # Dot velocity derived from consecutive frame timestamps — accurate even
        # when the loop sleeps longer than ALIGN_LOOP_DT due to network jitter,
        # because we use the actual camera frame clock, not perf_counter.
        if seg.ts_prev is not None:
            t_prev, pos_prev = seg.ts_prev
            dt_frame = t_now - t_prev
            vel = (pos - pos_prev) / dt_frame if dt_frame > 0.005 else np.zeros(2)
        else:
            vel = np.zeros(2)
        seg.ts_prev = (t_now, pos.copy())

        x_st  = np.array([err[0], err[1], vel[0], vel[1]])
        K     = _lqr_gain(seg.jac)
        w_lqr = np.clip(-K @ x_st, -ALIGN_MAX_SPEED, ALIGN_MAX_SPEED)

        # Shape the command in *dot-velocity* space so both limits are enforced on
        # what the dot actually does: fade out motion heading into the border, then
        # cap the overall speed. The Jacobian maps back to servo velocities.
        v_des    = seg.jac @ w_lqr
        v_des, _ = _border_guard(pos, v_des)
        w        = _servo_cmd_for(seg.jac, _cap_dot_speed(v_des))

        seg.servo_velocity(w)
        seg.moving = True
        sleep(ALIGN_LOOP_DT)

        # ── Phase 3: runaway guard + LMS Jacobian update ─────────────────────
        # Both use the same observation: where the dot actually went this period.
        ts_after = seg.get_dot_timestamped()
        if ts_after is not None:
            t_after, pos_after = ts_after
            dt_lms = t_after - t_now
            disp   = float(np.linalg.norm(pos_after - pos))
            ww     = float(w @ w)
            if dt_lms > 0.005:
                v_obs = (pos_after - pos) / dt_lms

                # Runaway guard: a wrong-sign or noise-fitted Jacobian sends the
                # dot away from the target at full speed until it leaves the
                # frame. If the dot's actual motion opposes what we asked for —
                # or the error simply keeps growing — for ALIGN_RUNAWAY_N periods
                # in a row, stop trusting the model and re-identify from scratch.
                v_mag = float(np.linalg.norm(v_obs))
                d_mag = float(np.linalg.norm(v_des))
                if v_mag >= ALIGN_RUNAWAY_V and d_mag > 1e-6:
                    wrong_way  = float(np.dot(v_obs, v_des)) < 0.0
                    err_after  = float(np.linalg.norm(
                        np.array(seg.target, dtype=float) - pos_after))
                    diverging  = err_after > err_mag
                    seg.bad = seg.bad + 1 if (wrong_way and diverging) else 0

                if seg.bad >= ALIGN_RUNAWAY_N:
                    seg.halt()
                    seg.fault = ("dot moved against the commanded direction for "
                                 f"{seg.bad} periods — Jacobian discarded, re-identifying")
                    print(f"[align:{seg.name}] runaway detected: {seg.fault}")
                    seg.unbind()
                    sleep(ALIGN_SETTLE_DT)
                    continue

                if disp >= LMS_GATE and ww > 1e-9:
                    residual = v_obs - seg.jac @ w
                    seg.jac += ALIGN_LMS_RATE * np.outer(residual, w) / ww


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
    data["servo_online"] = any(s.online for s in SEGMENTS)
    data["segments"]     = [s.state_dict() for s in SEGMENTS]
    data["align"]        = _align_state()
    data["camera"] = {
        "gain":        round(float(_cam_gain), 2),
        "gain_min":    CAM_GAIN_MIN,
        "gain_max":    CAM_GAIN_MAX,
        "exposure_us": _cam_exposure_us,
        "exp_min":     CAM_EXP_MIN,
        "exp_max":     CAM_EXP_MAX,
        "contrast":    round(float(_cam_contrast), 2),
        "brightness":  round(float(_cam_brightness), 2),
        "ae":          bool(_cam_ae),
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
    """Proxy a servo command to a segment's servo Pi.

    POST {"command": "tip+", "segment": "s0"}  — 'segment' optional, defaults to
    the first segment for backward compatibility with the single-mirror UI.
    """
    body = request.get_json(silent=True) or {}
    cmd  = str(body.get("command", "")).strip().lower()
    seg  = _segment_for(body)
    try:
        r = _requests.post(f"{seg.url}/servo",
                           json={"command": cmd}, timeout=1.5)
        seg.online = r.ok
        return jsonify(r.json()), r.status_code
    except Exception as exc:
        seg.online = False
        return jsonify({"error": str(exc)}), 503


@app.route("/servo/trim", methods=["GET", "POST"])
def servo_trim():
    """Proxy trim GET/POST to a segment's servo Pi.

    Target segment via ?segment=s1 (GET) or {"segment": "s1"} (POST); defaults
    to the first segment.
    """
    if request.method == "GET":
        seg = SEGMENTS_BY_NAME.get(request.args.get("segment", ""), SEGMENTS[0])
    else:
        seg = _segment_for(request.get_json(silent=True) or {})
    try:
        if request.method == "GET":
            r = _requests.get(f"{seg.url}/servo/trim", timeout=1.5)
        else:
            r = _requests.post(f"{seg.url}/servo/trim",
                               json=request.get_json(silent=True) or {},
                               timeout=1.5)
        seg.online = r.ok
        return jsonify(r.json()), r.status_code
    except Exception as exc:
        seg.online = False
        return jsonify({"error": str(exc)}), 503


@app.route("/servo/calibrate", methods=["POST"])
def servo_calibrate():
    """Proxy a calibrate (set + persist stop values) to a segment's servo Pi.

    POST {"segment": "s1", "stop": {"A": .., "B": .., "C": ..}}
    """
    body = request.get_json(silent=True) or {}
    seg  = _segment_for(body)
    try:
        r = _requests.post(f"{seg.url}/servo/calibrate", json=body, timeout=3.0)
        seg.online = r.ok
        return jsonify(r.json()), r.status_code
    except Exception as exc:
        seg.online = False
        return jsonify({"error": str(exc)}), 503


@app.route("/servo/auto", methods=["POST"])
def servo_auto():
    """Enable or toggle the auto-alignment controller."""
    global _auto_align
    body        = request.get_json(silent=True) or {}
    _auto_align = bool(body.get("enabled", not _auto_align))
    return jsonify({"auto_align": _auto_align})


def _align_state() -> Dict[str, Any]:
    """Master switch + shared tuning + per-segment states.

    Legacy top-level keys (dot/servos/target/identified/known/jacobian) mirror
    the first segment so the existing single-mirror UI keeps working unchanged.
    """
    first = SEGMENTS[0].state_dict()
    bx0, by0, bx1, by1 = safe_box()
    return {
        "enabled":   _auto_align,                     # master switch (all segments)
        "align_r":   round(float(ALIGN_R), 4),
        "lms_rate":  round(float(ALIGN_LMS_RATE), 4),
        "max_dot_speed": round(float(ALIGN_MAX_DOT_SPEED), 1),
        "probe_speed":   round(float(ALIGN_PROBE_SPEED), 3),
        "border_frac":   round(float(BORDER_FRAC), 3),
        "safe_box":      [round(bx0, 1), round(by0, 1), round(bx1, 1), round(by1, 1)],
        "segments":  [s.state_dict() for s in SEGMENTS],
        # legacy single-mirror fields (first segment)
        "dot":        first["dot"],
        "servos":     first["servos"],
        "target":     first["target"],
        "identified": first["identified"],
        "known":      first["known"],
        "jacobian":   first["jacobian"],
    }


@app.route("/align/config", methods=["GET", "POST"])
def align_config():
    """Configure the align controllers (one per segment/mirror).

    POST JSON (any subset):
      enabled      bool     — master switch: run/stop ALL enabled segments
      segment      "s0"     — which segment the per-segment fields apply to
                              (defaults to the first segment)
      seg_enabled  bool     — enable/disable that one segment
      dot          "A"/"B"  — which tracked dot that segment drives (re-identifies)
      target_x     float    — absolute target x (px) for that segment's dot
      target_y     float    — absolute target y (px)
      target_dx    float    — nudge target x (px)
      target_dy    float    — nudge target y (px)
      reidentify   bool     — discard that segment's Jacobian and re-probe
      align_r      float    — LQR control-effort weight (shared by all segments)
      lms_rate     float    — Jacobian adaptation rate (shared by all segments)
      max_dot_speed float   — px/s cap on how fast any dot is driven (shared)
      probe_speed  float    — servo velocity used during Jacobian probing (shared);
                              lower = smaller calibration steps

    Switching a segment's dot also forces re-identification of that segment.  If
    another segment already drives the requested dot the two swap dots, so the
    mirrors never both chase the same spot.
    """
    global _auto_align, ALIGN_R, ALIGN_LMS_RATE, ALIGN_MAX_DOT_SPEED, ALIGN_PROBE_SPEED
    if request.method == "GET":
        return jsonify(_align_state())

    body = request.get_json(silent=True) or {}

    if "enabled" in body:
        _auto_align = bool(body["enabled"])

    seg = _segment_for(body)

    if "seg_enabled" in body:
        seg.enabled = bool(body["seg_enabled"])

    # A manually-set dot is only a *hint* now: the identification probe decides
    # which dot this mirror really drives, so a wrong choice here can no longer
    # send the mirror chasing another mirror's dot.
    if "dot" in body:
        d = str(body["dot"]).upper()
        if d in ("A", "B") and d != seg.dot_hint:
            seg.dot_hint = d
            seg.unbind()                         # re-discover the binding by probing

    if body.get("reidentify"):
        seg.unbind()

    # Targets are clamped into the safe box — the controller will not drive a dot
    # to within BORDER_FRAC of the frame edge, so an out-of-bounds target would
    # just leave it pinned against the border.
    tx, ty = seg.target[0], seg.target[1]
    if "target_x"  in body: tx = float(body["target_x"])
    if "target_y"  in body: ty = float(body["target_y"])
    if "target_dx" in body: tx += float(body["target_dx"])
    if "target_dy" in body: ty += float(body["target_dy"])
    seg.target[0], seg.target[1] = clamp_to_safe(tx, ty)

    if "align_r" in body:
        ALIGN_R = float(np.clip(float(body["align_r"]), 0.001, 10.0))
    if "lms_rate" in body:
        ALIGN_LMS_RATE = float(np.clip(float(body["lms_rate"]), 0.0, 0.5))
    if "max_dot_speed" in body:
        ALIGN_MAX_DOT_SPEED = float(np.clip(float(body["max_dot_speed"]), 2.0, 300.0))
    if "probe_speed" in body:
        ALIGN_PROBE_SPEED = float(np.clip(float(body["probe_speed"]), 0.02, ALIGN_MAX_SPEED))

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


@app.route("/camera", methods=["POST"])
def camera_set():
    """Live image controls, applied by the capture thread (which owns picam2).

    POST any of:
      reset       true   → normal, well-exposed greyscale image (auto-exposure,
                           normal contrast) for spotting things by eye
      ae          bool   → auto-exposure on/off
      contrast    float  → 0..4   (1.0 = normal, 2.0 = dot-popping)
      brightness  float  → -1..1
      gain        float  → analogue gain (forces manual exposure / AE off)
    """
    global _cam_ctrl_req, _cam_contrast, _cam_brightness, _cam_ae, _cam_gain
    body  = request.get_json(silent=True) or {}
    ctrls = {}

    if body.get("reset"):
        _cam_ae, _cam_contrast, _cam_brightness = True, 1.0, 0.0
        ctrls.update({"AeEnable": True, "Contrast": 1.0,
                      "Brightness": 0.0, "Saturation": 0.0})
    else:
        if "ae" in body:
            _cam_ae = bool(body["ae"])
            ctrls["AeEnable"] = _cam_ae
            if not _cam_ae and _cam_exposure_us:
                ctrls["ExposureTime"] = int(_cam_exposure_us)
        if "contrast" in body:
            _cam_contrast = float(np.clip(float(body["contrast"]), 0.0, 4.0))
            ctrls["Contrast"] = _cam_contrast
        if "brightness" in body:
            _cam_brightness = float(np.clip(float(body["brightness"]), -1.0, 1.0))
            ctrls["Brightness"] = _cam_brightness
        if "gain" in body:
            _cam_gain = float(np.clip(float(body["gain"]), CAM_GAIN_MIN, CAM_GAIN_MAX))
            _cam_ae = False
            ctrls["AeEnable"] = False
            ctrls["AnalogueGain"] = _cam_gain

    _cam_ctrl_req = {**_cam_ctrl_req, **ctrls}   # merge; applied by capture thread
    return jsonify({"ok": True, "ae": _cam_ae,
                    "contrast": round(_cam_contrast, 2),
                    "brightness": round(_cam_brightness, 2),
                    "gain": round(float(_cam_gain), 2)})


# ─────────────────────────── Entry point ─────────────────────────────────────

if __name__ == "__main__":
    threading.Thread(target=capture_loop, daemon=True).start()
    # One independent align controller per segment (mirror → its dot).
    for _seg in SEGMENTS:
        threading.Thread(target=segment_align_loop, args=(_seg,),
                         daemon=True, name=f"align-{_seg.name}").start()
    print(f"HTTP server on port {WEB_PORT}")
    for _seg in SEGMENTS:
        print(f"Segment {_seg.name}: dot {_seg.dot} → {_seg.url} "
              f"(servos {','.join(_seg.servos)}, {'enabled' if _seg.enabled else 'disabled'})")
    app.run(host="0.0.0.0", port=WEB_PORT, threaded=True)
