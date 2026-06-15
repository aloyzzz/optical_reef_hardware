"""
comms_test_master.py  —  run this on the MASTER (detector) Pi

Sends a text message to the slave Pi and prints the reply.
Repeats every second so you can verify the link is stable.

Usage:
    python3 comms_test_master.py [slave-ip]

    slave-ip defaults to "servo-pi.local" (mDNS hostname).
    Replace with the slave Pi's IP address if mDNS is not available,
    e.g.:  python3 comms_test_master.py 192.168.1.42
"""

import sys
import time
import requests

SLAVE_IP   = sys.argv[1] if len(sys.argv) > 1 else "servo-pi.local"
SLAVE_URL  = f"http://{SLAVE_IP}:5000"
INTERVAL   = 1.0   # seconds between messages

def ping():
    """Check the slave is reachable before starting the loop."""
    try:
        r = requests.get(f"{SLAVE_URL}/ping", timeout=3)
        print(f"[master] ping → {r.json()}")
        return True
    except Exception as e:
        print(f"[master] ping failed: {e}")
        return False

def send(text):
    r = requests.post(
        f"{SLAVE_URL}/message",
        json={"msg": text},
        timeout=3,
    )
    return r.json()

if __name__ == "__main__":
    print(f"Target: {SLAVE_URL}")

    if not ping():
        print("Cannot reach slave Pi — check IP/hostname and that comms_test_slave.py is running.")
        sys.exit(1)

    print("Link OK. Sending messages every second (Ctrl-C to stop).\n")

    count = 0
    while True:
        count += 1
        msg = f"hello from master (msg #{count})"
        print(f"[master] sending:  {msg!r}")
        try:
            reply = send(msg)
            print(f"[master] received: {reply.get('msg')!r}")
        except Exception as e:
            print(f"[master] error: {e}")
        print()
        time.sleep(INTERVAL)
