"""
Servo control server — runs on the slave (mirror) Pi.

Start with:  python3 servo_server.py
Listens on:  http://0.0.0.0:5000

Routes
------
GET  /health          → {"status": "ok", "gpio": true/false}
POST /servo           → {"command": "x+"}  →  {"ok": true, "command": "x+"}
POST /servo/move      → {"moves": {"A": 0.4, "B": -1.2}}  proportional, smooth move
POST /servo/velocity  → {"velocities": {"A": 0.12}}  continuous velocity (watchdog-stopped)

Supported commands (POST /servo)
--------------------------------
  x+  x-   y+  y-   pist+  pist-   (composite mirror motions)
  a+    a-     b+     b-      c+     c-       (individual screws, fixed jog)

Smooth control (POST /servo/move)
---------------------------------
  Signed per-servo "amounts" in jog units (1.0 == one nominal jog), fractional
  allowed.  Each servo runs for |amount|·UNIT_MOVE_TIME then stops; all servos
  start together.  This gives the closed-loop tracker sub-jog resolution so its
  continuous LQR command isn't thrown away by integer-jog quantisation.

GPIO pins: A=19, B=20, C=21  (FS90R continuous-rotation servos)
"""

import threading
import os
import json
import signal
import atexit
from time import sleep, monotonic
from flask import Flask, request, jsonify

app = Flask(__name__)
_servo_lock = threading.Lock()

# Velocity-streaming state: the tracker pushes continuous per-servo velocities
# and the servos keep running until the next update.  A watchdog stops them if
# updates stop arriving (e.g. the tracker dies), so they can never run away.
VEL_WATCHDOG   = 0.4    # seconds without a velocity update before auto-stop
_vel_last_time = 0.0
_vel_active    = False


# ── GPIO / servo setup ────────────────────────────────────────────────────────

PIN_A = 19
PIN_B = 20
PIN_C = 21

# Per-servo "stop" (neutral) value. Each continuous-rotation servo has a slightly
# different true neutral, so these are calibrated per Pi and PERSISTED to disk:
# defaults below are just a guess; the real values are loaded from STOP_FILE on
# startup (if it exists) and written there by POST /servo/calibrate — so a
# calibration survives restarts.  1390us -> (1390-1500)/1000 = -0.11.
STOP = {"A": -0.11, "B": -0.11, "C": -0.11}
STOP_FILE = os.path.expanduser("~/.servo_stop.json")


def _load_stop():
    """Load persisted per-servo stop values from STOP_FILE (if present)."""
    try:
        with open(STOP_FILE) as f:
            data = json.load(f)
        for k in ("A", "B", "C"):
            if k in data:
                STOP[k] = max(-1.0, min(1.0, float(data[k])))
        print(f"Loaded servo stops from {STOP_FILE}: {STOP}")
    except FileNotFoundError:
        pass
    except Exception as _e:
        print(f"[WARN] could not load {STOP_FILE}: {_e}")


def _save_stop():
    """Persist the current per-servo stop values so they survive a restart."""
    try:
        with open(STOP_FILE, "w") as f:
            json.dump(dict(STOP), f)
        print(f"Saved servo stops to {STOP_FILE}: {STOP}")
        return True
    except Exception as _e:
        print(f"[WARN] could not save {STOP_FILE}: {_e}")
        return False


_load_stop()
DEFAULT_SPEED = 0.18
DEFAULT_TIME  = 0.15

# Smooth-move scaling: a move "amount" of 1.0 == one nominal jog == this many
# seconds of travel at DEFAULT_SPEED.  Fractional amounts run proportionally
# shorter, giving sub-jog resolution.  MAX_MOVE_TIME caps any single command.
UNIT_MOVE_TIME = DEFAULT_TIME
MAX_MOVE_TIME  = 1.2

PINS = {"A": PIN_A, "B": PIN_B, "C": PIN_C}

# value (-1..+1) → servo pulse width in microseconds. 0 -> 1500us (neutral/stop),
# matching the gpiozero range min=0.5ms / max=2.5ms used previously and the
# servo_diag.py sweep. 1000us per unit.
_CENTER_US = 1500
_US_PER_UNIT = 1000
_MIN_US, _MAX_US = 500, 2500

# Drive pigpio directly (set_servo_pulsewidth) instead of through gpiozero.
# gpiozero's Servo wrapper was producing intermittent full-speed runaway even
# though raw pigpio holds these exact pins rock-steady (proven via servo_diag.py).
# pigpio generates PWM via DMA — immune to CPU load spikes from OpenCV/Flask.
# Requires pigpiod running: sudo pigpiod  (sudo systemctl enable pigpiod)
try:
    import pigpio

    _pi = pigpio.pi()
    if not _pi.connected:
        raise RuntimeError("pigpiod not reachable — run 'sudo pigpiod'")
    GPIO_AVAILABLE = True
    print(f"GPIO servos initialised (raw pigpio DMA) on pins A={PIN_A}, B={PIN_B}, C={PIN_C}")
except Exception as _e:
    print(f"[WARN] GPIO/pigpio not available ({_e}) — servo commands will be logged only")
    print(f"[WARN] If on Pi: run 'sudo pigpiod' then restart this server.")
    _pi = None
    GPIO_AVAILABLE = False


def _clamp(v):
    return max(-1.0, min(1.0, v))


def _set(name, value):
    """Write a -1..+1 value to a servo as a clamped pulse width."""
    us = int(round(_CENTER_US + _clamp(value) * _US_PER_UNIT))
    us = max(_MIN_US, min(_MAX_US, us))
    _pi.set_servo_pulsewidth(PINS[name], us)


def _stop_all():
    for name in PINS:
        _set(name, STOP[name])


def _release_all():
    """Stop sending pulses on every servo pin (pulsewidth 0).

    pigpio keeps generating the last pulse *forever*, so if the server exits
    without doing this, the servos stay driven — e.g. a leftover velocity
    command keeps a continuous-rotation servo spinning with no server running.
    Setting the pulse to 0 releases the line so nothing is driven once we're
    gone.  Safe to call when GPIO isn't available.
    """
    if _pi is None:
        return
    for pin in PINS.values():
        try:
            _pi.set_servo_pulsewidth(pin, 0)
        except Exception:
            pass


def _jog_motor(name, direction, duration=DEFAULT_TIME, speed=DEFAULT_SPEED):
    if not GPIO_AVAILABLE:
        print(f"  [DRY] jog {name} {'fwd' if direction > 0 else 'rev'}")
        return
    # try/finally guarantees the servo is returned to stop even if the move
    # raises — otherwise it would keep running until the next command.
    try:
        _set(name, STOP[name] + direction * speed)
        sleep(duration)
    finally:
        _set(name, STOP[name])


def _jog_combo(weights, duration=DEFAULT_TIME, speed=DEFAULT_SPEED):
    if not GPIO_AVAILABLE:
        print(f"  [DRY] combo {weights}")
        return
    # try/finally guarantees all servos are stopped even if one set raises
    # mid-loop — otherwise the already-set servos keep running.
    try:
        for name, w in weights.items():
            _set(name, STOP[name] + w * speed)
        sleep(duration)
    finally:
        _stop_all()


def _set_velocity(vels):
    """Set continuous per-servo velocities (offset from STOP, in [-1, 1]).

    Returns immediately — the servos keep running at these velocities until the
    next update or until the watchdog stops them.  vels: {"A": v, ...}.
    """
    global _vel_last_time, _vel_active
    if not GPIO_AVAILABLE:
        print(f"  [DRY] velocity {vels}")
        _vel_last_time = monotonic()
        _vel_active    = any(abs(float(v)) > 1e-6 for v in vels.values())
        return
    with _servo_lock:
        for name, v in vels.items():
            if name in PINS:
                _set(name, STOP[name] + _clamp(float(v)))
        _vel_last_time = monotonic()
        _vel_active    = any(abs(float(v)) > 1e-6 for v in vels.values())


def _velocity_watchdog():
    """Stop the servos if velocity updates stop arriving (failsafe)."""
    while True:
        sleep(VEL_WATCHDOG / 2.0)
        global _vel_active
        if (GPIO_AVAILABLE and _vel_active
                and monotonic() - _vel_last_time > VEL_WATCHDOG):
            with _servo_lock:
                _stop_all()
            _vel_active = False
            print("[watchdog] velocity timeout — servos stopped")


def _timed_move(moves, speed=DEFAULT_SPEED):
    """Move several servos simultaneously by signed, proportional amounts.

    moves: {"A": amount, ...} in jog units (1.0 == UNIT_MOVE_TIME at `speed`).
    Fractional amounts give smooth sub-jog resolution.  All servos start
    together; each stops after |amount|·UNIT_MOVE_TIME (capped at MAX_MOVE_TIME).
    """
    if not GPIO_AVAILABLE:
        print(f"  [DRY] move {moves}")
        return
    speed = abs(float(speed))
    schedule = []          # (duration, name) for servos that actually move
    try:
        for name, amt in moves.items():
            if name not in PINS:
                continue
            amt = float(amt)
            dur = min(abs(amt) * UNIT_MOVE_TIME, MAX_MOVE_TIME)
            if dur <= 0.0 or speed <= 0.0:
                continue
            _set(name, STOP[name] + (1.0 if amt > 0 else -1.0) * speed)
            schedule.append((dur, name))
        # Stop each servo as its own duration elapses (all started together).
        schedule.sort()
        elapsed = 0.0
        for dur, name in schedule:
            sleep(max(0.0, dur - elapsed))
            elapsed = dur
            _set(name, STOP[name])
    finally:
        # Safety net — ensure every servo we commanded is back at stop.
        for _dur, name in schedule:
            _set(name, STOP[name])


# ── Command dispatch table ────────────────────────────────────────────────────

_COMMANDS = {
    # Mount layout: B top-left, A top-right, C below A.  x and y are decoupled —
    # the servo not in the moving pair matches A to null the other axis.
    #   x = tilt about y-axis (moves the spot along x): A vs B, C follows A
    #   y = tilt about x-axis (moves the spot along y): A vs C, B follows A
    # Signs are a starting guess; flip a pair if a direction is reversed.
    "x+":    lambda: _jog_combo({"A":  1, "B": -1, "C":  1}),
    "x-":    lambda: _jog_combo({"A": -1, "B":  1, "C": -1}),
    "y+":    lambda: _jog_combo({"A": -1, "B": -1, "C":  1}),   # flipped: y was reversed
    "y-":    lambda: _jog_combo({"A":  1, "B":  1, "C": -1}),
    "pist+": lambda: _jog_combo({"A":  1, "B":  1, "C":  1}),   # all three together
    "pist-": lambda: _jog_combo({"A": -1, "B": -1, "C": -1}),
    "a+":    lambda: _jog_motor("A",  1),
    "a-":    lambda: _jog_motor("A", -1),
    "b+":    lambda: _jog_motor("B",  1),
    "b-":    lambda: _jog_motor("B", -1),
    "c+":    lambda: _jog_motor("C",  1),
    "c-":    lambda: _jog_motor("C", -1),
}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok", "gpio": GPIO_AVAILABLE,
                    "stop": dict(STOP)})


@app.route("/servo", methods=["POST"])
def servo():
    body = request.get_json(silent=True) or {}
    cmd  = str(body.get("command", "")).strip().lower()
    if cmd not in _COMMANDS:
        return jsonify({"error": f"unknown command: {cmd!r}"}), 400
    # Execute synchronously under the lock — blocks until the jog completes.
    # An async queue lets commands pile up faster than they execute, causing
    # the servo to appear to run at full speed continuously.
    with _servo_lock:
        try:
            _COMMANDS[cmd]()
            print(f"servo: {cmd}")
        except Exception as _e:
            print(f"[servo] {cmd}: {_e}")
            return jsonify({"error": str(_e)}), 500
    return jsonify({"ok": True, "command": cmd})


@app.route("/servo/move", methods=["POST"])
def servo_move():
    """Proportional, simultaneous multi-servo move for smooth closed-loop control.

    POST {"moves": {"A": 0.4, "B": -1.2, "C": 0.0}, "speed": 0.18}
      - amounts are signed jog units (fractional allowed; 1.0 == one nominal jog)
      - "speed" is optional (defaults to DEFAULT_SPEED)
    Runs synchronously under the servo lock and returns once the move completes.
    """
    body  = request.get_json(silent=True) or {}
    moves = body.get("moves")
    if not isinstance(moves, dict) or not moves:
        return jsonify({"error": "provide 'moves': {servo: amount}"}), 400

    parsed = {}
    for k, v in moves.items():
        name = str(k).upper()
        if name not in PINS:
            continue
        try:
            parsed[name] = float(v)
        except (TypeError, ValueError):
            return jsonify({"error": f"bad amount for {name!r}"}), 400
    if not parsed:
        return jsonify({"error": "no valid servos in 'moves'"}), 400

    try:
        speed = float(body.get("speed", DEFAULT_SPEED))
    except (TypeError, ValueError):
        return jsonify({"error": "bad 'speed'"}), 400

    with _servo_lock:
        try:
            _timed_move(parsed, speed)
            print(f"move: { {k: round(v, 3) for k, v in parsed.items()} }")
        except Exception as _e:
            print(f"[move] {_e}")
            return jsonify({"error": str(_e)}), 500
    return jsonify({"ok": True, "moves": parsed})


@app.route("/servo/velocity", methods=["POST"])
def servo_velocity():
    """Continuous velocity streaming for smooth closed-loop control.

    POST {"velocities": {"A": 0.12, "B": -0.30, "C": 0.0}}
      - each value is a signed velocity offset from STOP, clamped to [-1, 1]
      - servos keep running at these velocities until the next update
      - send all zeros (or stop updating) to halt; the watchdog also halts them
        VEL_WATCHDOG seconds after the last update
    Returns immediately (does not block while the servos move).
    """
    body = request.get_json(silent=True) or {}
    vels = body.get("velocities")
    if not isinstance(vels, dict) or not vels:
        return jsonify({"error": "provide 'velocities': {servo: value}"}), 400

    parsed = {}
    for k, v in vels.items():
        name = str(k).upper()
        if name not in PINS:
            continue
        try:
            parsed[name] = float(v)
        except (TypeError, ValueError):
            return jsonify({"error": f"bad velocity for {name!r}"}), 400
    if not parsed:
        return jsonify({"error": "no valid servos in 'velocities'"}), 400

    try:
        _set_velocity(parsed)
    except Exception as _e:
        print(f"[velocity] {_e}")
        return jsonify({"error": str(_e)}), 500
    return jsonify({"ok": True, "velocities": parsed})


@app.route("/servo/trim", methods=["GET", "POST"])
def servo_trim():
    """GET  → current STOP values.
       POST → {"servo": "A", "delta": 0.001}  nudge that servo's stop value,
              apply it live so you can hear/see the result immediately.
              {"servo": "A", "value": 0.012}  set absolute value.
    """
    if request.method == "GET":
        return jsonify({"stop": dict(STOP)})

    body  = request.get_json(silent=True) or {}
    name  = str(body.get("servo", "")).upper()
    if name not in ("A", "B", "C"):
        return jsonify({"error": "servo must be A, B, or C"}), 400

    if "value" in body:
        new_val = float(body["value"])
    elif "delta" in body:
        new_val = STOP[name] + float(body["delta"])
    else:
        return jsonify({"error": "provide 'delta' or 'value'"}), 400

    new_val = max(-1.0, min(1.0, new_val))
    STOP[name] = round(new_val, 4)
    if GPIO_AVAILABLE:
        with _servo_lock:
            _set(name, STOP[name])
    print(f"trim: STOP[{name}] = {STOP[name]:.4f}")
    return jsonify({"stop": dict(STOP)})


@app.route("/servo/calibrate", methods=["POST"])
def servo_calibrate():
    """Set the per-servo stop (neutral) values and PERSIST them to disk.

    POST {"stop": {"A": -0.10, "B": -0.09, "C": -0.11}}   (any subset of A/B/C)

    Applies the new stops live (so drift stops immediately) and writes them to
    STOP_FILE, so they are reloaded automatically on the next startup.
    """
    body = request.get_json(silent=True) or {}
    stop = body.get("stop")
    if not isinstance(stop, dict) or not stop:
        return jsonify({"error": "provide 'stop': {A,B,C}"}), 400

    for k, v in stop.items():
        name = str(k).upper()
        if name not in STOP:
            continue
        try:
            STOP[name] = round(max(-1.0, min(1.0, float(v))), 4)
        except (TypeError, ValueError):
            return jsonify({"error": f"bad value for {name!r}"}), 400

    if GPIO_AVAILABLE:
        with _servo_lock:
            _stop_all()          # apply the new neutrals right away
    saved = _save_stop()
    return jsonify({"stop": dict(STOP), "saved": saved})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if GPIO_AVAILABLE:
        # Fresh start: clear any stale pulse a previously-killed server left on
        # the pins (pigpio persists them), then hold the calibrated stop.  This
        # is the recovery path when the last server was hard-killed (e.g. the
        # terminal was closed) and couldn't clean up after itself.
        _release_all()
        _stop_all()

    # Release the pins on exit so a stopped server never leaves a servo driven.
    # Covers Ctrl-C (SIGINT), kill (SIGTERM) and terminal-close (SIGHUP).  A hard
    # SIGKILL can't be caught — the startup reset above recovers from that case.
    atexit.register(_release_all)

    def _shutdown(_signum, _frame):
        _release_all()
        os._exit(0)

    for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(_sig, _shutdown)

    threading.Thread(target=_velocity_watchdog, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True)
