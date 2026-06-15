from gpiozero import Servo
from time import sleep

# -----------------------------
# GPIO PINS
# -----------------------------
# Change these if your wiring is different.
PIN_A = 19   # actuator / screw A
PIN_B = 20   # actuator / screw B
PIN_C = 21   # actuator / screw C

# -----------------------------
# SERVO SETUP
# -----------------------------
# FS90R continuous servos.
# You may need to tune these stop values.
sA = Servo(PIN_A, min_pulse_width=0.0005, max_pulse_width=0.0025, frame_width=0.02)
sB = Servo(PIN_B, min_pulse_width=0.0005, max_pulse_width=0.0025, frame_width=0.02)
sC = Servo(PIN_C, min_pulse_width=0.0005, max_pulse_width=0.0025, frame_width=0.02)

servos = {
    "A": sA,
    "B": sB,
    "C": sC,
}

# If a servo creeps when stopped, adjust these.
STOP = {
    "A": 0.00,
    "B": 0.00,
    "C": 0.00,
}

DEFAULT_SPEED = 0.18
DEFAULT_TIME = 0.15


def stop_all():
    for name, servo in servos.items():
        servo.value = STOP[name]


def jog_motor(name, direction, duration=DEFAULT_TIME, speed=DEFAULT_SPEED):
    """
    direction = +1 or -1
    """
    servo = servos[name]
    servo.value = STOP[name] + direction * speed
    sleep(duration)
    servo.value = STOP[name]


def jog_combo(weights, duration=DEFAULT_TIME, speed=DEFAULT_SPEED):
    """
    weights example:
    {"A": 1, "B": -1, "C": 0}
    """
    for name, weight in weights.items():
        servos[name].value = STOP[name] + weight * speed

    sleep(duration)
    stop_all()


def trim_interactive():
    """Send live PWM to one servo so you can dial in its STOP value.

    Run this, watch the servo, and nudge the value until it truly stops.
    Press Enter to save the result to the STOP dict for this session.
    """
    print("\nTrim calibration — find the true neutral for each servo.")
    print("Enter a servo name (A/B/C) or 'done' to exit.\n")
    while True:
        name = input("servo> ").strip().upper()
        if name == "DONE":
            break
        if name not in servos:
            print("Unknown servo. Try A, B, or C.")
            continue
        current = STOP[name]
        print(f"  Servo {name} — current STOP={current:.4f}")
        print("  Commands: +0.01 / -0.01 / +0.001 / -0.001 / save / cancel")
        servos[name].value = current
        while True:
            cmd = input(f"  [{current:+.4f}]> ").strip().lower()
            if cmd in ("+0.01", "+"):
                current = round(current + 0.01, 4)
            elif cmd in ("-0.01", "-"):
                current = round(current - 0.01, 4)
            elif cmd == "+0.001":
                current = round(current + 0.001, 4)
            elif cmd == "-0.001":
                current = round(current - 0.001, 4)
            elif cmd == "save":
                STOP[name] = current
                servos[name].detach()
                print(f"  Saved STOP['{name}'] = {current:.4f}\n")
                break
            elif cmd == "cancel":
                servos[name].detach()
                break
            else:
                print("  +0.01 / -0.01 / +0.001 / -0.001 / save / cancel")
                continue
            current = max(-1.0, min(1.0, current))
            servos[name].value = current
            print(f"  → {current:+.4f}")


# -----------------------------
# MIRROR MOTION DEFINITIONS
# -----------------------------
# These mappings are starting guesses.
# You will likely need to flip signs after testing.
def tip_plus():
    jog_combo({"A": 1, "B": -1, "C": 0})


def tip_minus():
    jog_combo({"A": -1, "B": 1, "C": 0})


def tilt_plus():
    jog_combo({"A": 1, "B": 1, "C": -1})


def tilt_minus():
    jog_combo({"A": -1, "B": -1, "C": 1})


def piston_plus():
    jog_combo({"A": 1, "B": 1, "C": 1})


def piston_minus():
    jog_combo({"A": -1, "B": -1, "C": -1})


def print_help():
    print()
    print("Commands:")
    print("  a+       move screw A forward")
    print("  a-       move screw A reverse")
    print("  b+       move screw B forward")
    print("  b-       move screw B reverse")
    print("  c+       move screw C forward")
    print("  c-       move screw C reverse")
    print()
    print("  tip+     tip mirror one way")
    print("  tip-     tip mirror opposite way")
    print("  tilt+    tilt mirror one way")
    print("  tilt-    tilt mirror opposite way")
    print("  pist+    piston mirror forward")
    print("  pist-    piston mirror backward")
    print()
    print("  stop     stop all servos")
    print("  trim     calibrate stop-neutral per servo")
    print("  help     show this menu")
    print("  q        quit")
    print()


def main():
    stop_all()
    print_help()

    try:
        while True:
            cmd = input("mirror> ").strip().lower()

            if cmd == "a+":
                jog_motor("A", 1)
            elif cmd == "a-":
                jog_motor("A", -1)

            elif cmd == "b+":
                jog_motor("B", 1)
            elif cmd == "b-":
                jog_motor("B", -1)

            elif cmd == "c+":
                jog_motor("C", 1)
            elif cmd == "c-":
                jog_motor("C", -1)

            elif cmd == "tip+":
                tip_plus()
            elif cmd == "tip-":
                tip_minus()

            elif cmd == "tilt+":
                tilt_plus()
            elif cmd == "tilt-":
                tilt_minus()

            elif cmd in ["pist+", "piston+"]:
                piston_plus()
            elif cmd in ["pist-", "piston-"]:
                piston_minus()

            elif cmd == "stop":
                stop_all()

            elif cmd == "trim":
                trim_interactive()

            elif cmd == "help":
                print_help()

            elif cmd in ["q", "quit", "exit"]:
                break

            else:
                print("Unknown command. Type help.")

    finally:
        stop_all()
        print("Exited safely.")


if __name__ == "__main__":
    main()