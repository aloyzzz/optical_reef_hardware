"""
Servo control server — runs on the slave (mirror) Pi.

Start with:  python3 servo_server.py
Listens on:  http://0.0.0.0:5000

Routes
------
GET  /health          → {"status": "ok", "gpio": true/false}
POST /servo           → {"command": "tip+"}  →  {"ok": true, "command": "tip+"}

Supported commands
------------------
  tip+  tip-   tilt+  tilt-   pist+  pist-   (composite mirror motions)
  a+    a-     b+     b-      c+     c-       (individual screws)

GPIO pins: A=18, B=19, C=20  (FS90R continuous-rotation servos)
"""

import threading
from time import sleep
from flask import Flask, request, jsonify

app = Flask(__name__)
_servo_lock = threading.Lock()


# ── GPIO / servo setup ────────────────────────────────────────────────────────

PIN_A = 19
PIN_B = 20
PIN_C = 21

STOP = {"A": 0.0, "B": 0.0, "C": 0.0}   # tune per servo to eliminate creep
DEFAULT_SPEED = 0.18
DEFAULT_TIME  = 0.15

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


# ── Command dispatch table ────────────────────────────────────────────────────

_COMMANDS = {
    "tip+":  lambda: _jog_combo({"A":  1, "B": -1, "C":  0}),
    "tip-":  lambda: _jog_combo({"A": -1, "B":  1, "C":  0}),
    "tilt+": lambda: _jog_combo({"A":  1, "B":  1, "C": -1}),
    "tilt-": lambda: _jog_combo({"A": -1, "B": -1, "C":  1}),
    "pist+": lambda: _jog_combo({"A":  1, "B":  1, "C":  1}),
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


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if GPIO_AVAILABLE:
        _stop_all()
    app.run(host="0.0.0.0", port=5000, threaded=True)
