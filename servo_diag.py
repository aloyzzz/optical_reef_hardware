"""
Servo diagnostic — bypasses Flask, talks straight to pigpio.

Run on the Pi:  python3 servo_diag.py

It does three things:
  1. Confirms pigpiod is reachable and reports the daemon version.
  2. Holds each servo at the "stop" pulse (1500 us) so you can see whether a
     stopped servo actually stops. If it spins here, the problem is the neutral
     pulse / wiring / power — NOT the server logic.
  3. Lets you sweep the pulse width live to find each servo's true stop point.

Controls (after it starts):
  a / b / c   select servo A / B / C
  + / -       nudge pulse width by 10 us
  ] / [       nudge by 1 us (fine)
  s           snap selected servo to 1500 us (nominal stop)
  0           set ALL servos to their current stop and hold
  q           quit (sets all to 0 = signal off)
"""

import sys
import pigpio

PINS = {"A": 19, "B": 20, "C": 21}   # BCM — must match servo_server.py
NOMINAL_STOP_US = 1500
MIN_US, MAX_US = 500, 2500

pi = pigpio.pi()                      # connects to local pigpiod
if not pi.connected:
    print("[FAIL] Could not connect to pigpiod. Run: sudo pigpiod")
    sys.exit(1)

print(f"pigpiod connected (version {pi.get_pigpio_version()})")
print(f"Pins: {PINS}")
print("Holding all servos at 1500 us (nominal stop) for 3 s — watch them.")
print("If any servo spins now, it is a neutral/wiring/power issue, not the code.\n")

for name, pin in PINS.items():
    pi.set_servo_pulsewidth(pin, NOMINAL_STOP_US)

import time
time.sleep(3)

# ── interactive sweep ─────────────────────────────────────────────────────────
try:
    import termios, tty

    def _getch():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
except Exception:
    _getch = lambda: input("key> ")[:1]

pw = {name: NOMINAL_STOP_US for name in PINS}
sel = "A"
print("Interactive mode. Select a/b/c, then +/- (10us) or ]/[ (1us). q to quit.\n")

while True:
    pin = PINS[sel]
    print(f"\r[{sel}] pulse = {pw[sel]:4d} us   ", end="", flush=True)
    k = _getch()
    if k in ("a", "b", "c"):
        sel = k.upper()
    elif k == "+":
        pw[sel] = min(MAX_US, pw[sel] + 10)
    elif k == "-":
        pw[sel] = max(MIN_US, pw[sel] - 10)
    elif k == "]":
        pw[sel] = min(MAX_US, pw[sel] + 1)
    elif k == "[":
        pw[sel] = max(MIN_US, pw[sel] - 1)
    elif k == "s":
        pw[sel] = NOMINAL_STOP_US
    elif k == "0":
        for n in PINS:
            pi.set_servo_pulsewidth(PINS[n], pw[n])
        continue
    elif k == "q":
        break
    else:
        continue
    pi.set_servo_pulsewidth(pin, pw[sel])

# ── shutdown: 0 = stop sending pulses ─────────────────────────────────────────
print("\n\nFinal stop pulses found (us):")
for name in PINS:
    print(f"  {name}: {pw[name]}  -> gpiozero value = {(pw[name]-1500)/1000:+.3f}")
for pin in PINS.values():
    pi.set_servo_pulsewidth(pin, 0)
pi.stop()
print("Signal off. Plug these gpiozero values into STOP in servo_server.py.")
