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

PIN_A = 18
PIN_B = 19
PIN_C = 20

STOP = {"A": 0.0, "B": 0.0, "C": 0.0}   # tune per servo to eliminate creep
DEFAULT_SPEED = 0.18
DEFAULT_TIME  = 0.15

try:
    from gpiozero import Servo as _GPIOServo

    _sA = _GPIOServo(PIN_A, min_pulse_width=0.0005, max_pulse_width=0.0025, frame_width=0.02)
    _sB = _GPIOServo(PIN_B, min_pulse_width=0.0005, max_pulse_width=0.0025, frame_width=0.02)
    _sC = _GPIOServo(PIN_C, min_pulse_width=0.0005, max_pulse_width=0.0025, frame_width=0.02)
    _servos = {"A": _sA, "B": _sB, "C": _sC}
    GPIO_AVAILABLE = True
    print("GPIO servos initialised on pins A=18, B=19, C=20")
except Exception as _e:
    print(f"[WARN] GPIO not available ({_e}) — servo commands will be logged only")
    _servos = {}
    GPIO_AVAILABLE = False


def _stop_all():
    for name, servo in _servos.items():
        servo.value = STOP[name]


def _jog_motor(name, direction, duration=DEFAULT_TIME, speed=DEFAULT_SPEED):
    if not GPIO_AVAILABLE:
        print(f"  [DRY] jog {name} {'fwd' if direction > 0 else 'rev'}")
        return
    _servos[name].value = STOP[name] + direction * speed
    sleep(duration)
    _servos[name].value = STOP[name]


def _jog_combo(weights, duration=DEFAULT_TIME, speed=DEFAULT_SPEED):
    if not GPIO_AVAILABLE:
        print(f"  [DRY] combo {weights}")
        return
    for name, w in weights.items():
        _servos[name].value = STOP[name] + w * speed
    sleep(duration)
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
    return jsonify({"status": "ok", "gpio": GPIO_AVAILABLE})


@app.route("/servo", methods=["POST"])
def servo():
    body = request.get_json(silent=True) or {}
    cmd  = str(body.get("command", "")).strip().lower()
    if cmd not in _COMMANDS:
        return jsonify({"error": f"unknown command: {cmd!r}"}), 400
    with _servo_lock:
        _COMMANDS[cmd]()
    print(f"servo: {cmd}")
    return jsonify({"ok": True, "command": cmd})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if GPIO_AVAILABLE:
        _stop_all()
    app.run(host="0.0.0.0", port=5000, threaded=True)
