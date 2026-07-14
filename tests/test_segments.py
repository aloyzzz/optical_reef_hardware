"""
Hardware-free functional tests for the multi-segment refactor.

Runs on the detector Pi's venv (needs the real picamera2/cv2/flask/scipy imports)
but touches NO camera and NO servos: it imports pi_disk_tracker, injects a fake
tracking state, and drives the Flask routes via app.test_client().  The align
threads only start under __main__, so importing never moves anything.

Run:  cd ~/optical_reef_hardware && myenv/bin/python tests/test_segments.py
Exit code 0 = all passed.
"""
import os
import sys

# Import the module under test (repo root is the parent of tests/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pi_disk_tracker as m   # noqa: E402

FAILS = []


def check(name, cond):
    print(("  PASS  " if cond else "  FAIL  ") + name)
    if not cond:
        FAILS.append(name)


def fake_state():
    return {
        "timestamp": 1000.0,
        "dots": [
            {"label": "A", "x": 10.0,  "y": 20.0,  "lost_frames": 0, "vel_x": 0.0, "vel_y": 0.0},
            {"label": "B", "x": 100.0, "y": 200.0, "lost_frames": 0, "vel_x": 0.0, "vel_y": 0.0},
        ],
    }


print("== segment configuration ==")
names = [s.name for s in m.SEGMENTS]
check("two segments configured", len(m.SEGMENTS) == 2)
check("segments are s0 and s1", names == ["s0", "s1"])
s0 = m.SEGMENTS_BY_NAME["s0"]
s1 = m.SEGMENTS_BY_NAME["s1"]
check("s0 drives dot A", s0.dot == "A")
check("s1 drives dot B", s1.dot == "B")
check("s0 enabled by default", s0.enabled is True)
check("s1 disabled until SERVO_PI_URL_1 set", s1.enabled is False)
check("s0 has 2 servos", s0.n == 2 and s0.servos == ("A", "B"))
check("jacobian starts unknown", s0.known == [False, False])

print("== dot lookup binds each mirror to its own dot ==")
m._tracking_state = fake_state()
r0 = s0.get_dot_timestamped()
r1 = s1.get_dot_timestamped()
check("s0 finds dot A at (10,20)", r0 is not None and list(r0[1]) == [10.0, 20.0])
check("s1 finds dot B at (100,200)", r1 is not None and list(r1[1]) == [100.0, 200.0])

# A lost dot returns None (controller will halt).
m._tracking_state = {"timestamp": 1.0,
                     "dots": [{"label": "A", "x": 1, "y": 2, "lost_frames": 3}]}
check("lost dot -> None", s0.get_dot_pos() is None)
check("missing dot -> None", s1.get_dot_pos() is None)
m._tracking_state = fake_state()

print("== segment selector ==")
check("_segment_for defaults to first", m._segment_for({}) is s0)
check("_segment_for picks named", m._segment_for({"segment": "s1"}) is s1)
check("_segment_for unknown -> first", m._segment_for({"segment": "zz"}) is s0)

print("== state_dict shape ==")
sd = s0.state_dict()
for k in ("name", "url", "dot", "servos", "enabled", "online",
          "target", "identified", "known", "jacobian"):
    check(f"state_dict has '{k}'", k in sd)

print("== Flask routes (test client, no server/camera) ==")
c = m.app.test_client()

j = c.get("/align/config").get_json()
check("GET /align/config lists 2 segments", len(j.get("segments", [])) == 2)
check("GET /align/config has master 'enabled'", "enabled" in j)

# Per-segment target set.
c.post("/align/config", json={"segment": "s1", "target_x": 300, "target_y": 150})
check("s1 target updated", s1.target == [300.0, 150.0])
check("s0 target untouched", s0.target != [300.0, 150.0])

# Per-segment enable toggle.
c.post("/align/config", json={"segment": "s1", "seg_enabled": True})
check("s1 seg_enabled -> True", s1.enabled is True)
c.post("/align/config", json={"segment": "s1", "seg_enabled": False})
check("s1 seg_enabled -> False", s1.enabled is False)

# Switching a segment's dot resets its Jacobian.
s0.known = [True, True]
c.post("/align/config", json={"segment": "s0", "dot": "B"})
check("changing dot resets Jacobian", s0.known == [False, False] and s0.dot == "B")
c.post("/align/config", json={"segment": "s0", "dot": "A"})   # restore

# Master switch flips but stays off at end (no motion).
c.post("/align/config", json={"enabled": True})
check("master switch enable", m._auto_align is True)
c.post("/align/config", json={"enabled": False})
check("master switch disable", m._auto_align is False)

# /state exposes the segments array and stays 200 with a fake tracking state.
rs = c.get("/state")
check("GET /state -> 200", rs.status_code == 200)
check("GET /state has segments[]", len(rs.get_json().get("segments", [])) == 2)

print()
if FAILS:
    print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
    sys.exit(1)
print("ALL PASSED")
sys.exit(0)
