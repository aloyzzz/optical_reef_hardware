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

try:
    from gpiozero import Servo as _GPIOServo
    from gpiozero.pins.pigpio import PiGPIOFactory as _PiGPIOFactory

    # pigpio generates PWM via DMA — immune to CPU load spikes from OpenCV/Flask.
    # Software PWM (gpiozero default) misses cycles under load, which the FS90R
    # interprets as a lost signal and responds with full-speed runaway.
    # Requires pigpiod running: sudo pigpiod  (sudo systemctl enable pigpiod)
    _factory = _PiGPIOFactory()
    _sA = _GPIOServo(PIN_A, min_pulse_width=0.0005, max_pulse_width=0.0025,
                     frame_width=0.02, pin_factory=_factory)
    _sB = _GPIOServo(PIN_B, min_pulse_width=0.0005, max_pulse_width=0.0025,
                     frame_width=0.02, pin_factory=_factory)
    _sC = _GPIOServo(PIN_C, min_pulse_width=0.0005, max_pulse_width=0.0025,
                     frame_width=0.02, pin_factory=_factory)
    _servos = {"A": _sA, "B": _sB, "C": _sC}
    GPIO_AVAILABLE = True
    print(f"GPIO servos initialised (pigpio DMA) on pins A={PIN_A}, B={PIN_B}, C={PIN_C}")
except Exception as _e:
    print(f"[WARN] GPIO/pigpio not available ({_e}) — servo commands will be logged only")
    print(f"[WARN] If on Pi: run 'sudo pigpiod' then restart this server.")
    _servos = {}
    GPIO_AVAILABLE = False


def _stop_all():
    for name, servo in _servos.items():
        servo.value = STOP[name]


def _clamp(v):
    return max(-1.0, min(1.0, v))


def _jog_motor(name, direction, duration=DEFAULT_TIME, speed=DEFAULT_SPEED):
    if not GPIO_AVAILABLE:
        print(f"  [DRY] jog {name} {'fwd' if direction > 0 else 'rev'}")
        return
    # try/finally guarantees the servo is stopped even if setting the value
    # raises (e.g. STOP+speed exceeds ±1.0) — otherwise it runs at full speed
    # until the next command.
    try:
        _servos[name].value = _clamp(STOP[name] + direction * speed)
        sleep(duration)
    finally:
        _servos[name].value = STOP[name]


def _jog_combo(weights, duration=DEFAULT_TIME, speed=DEFAULT_SPEED):
    if not GPIO_AVAILABLE:
        print(f"  [DRY] combo {weights}")
        return
    # try/finally guarantees all servos are stopped even if one value set
    # raises mid-loop — otherwise the already-set servos keep running.
    try:
        for name, w in weights.items():
            _servos[name].value = _clamp(STOP[name] + w * speed)
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
            _servos[name].value = STOP[name]
    print(f"trim: STOP[{name}] = {STOP[name]:.4f}")
    return jsonify({"stop": dict(STOP)})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if GPIO_AVAILABLE:
        _stop_all()
    app.run(host="0.0.0.0", port=5000, threaded=True)
