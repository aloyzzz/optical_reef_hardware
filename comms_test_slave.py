"""
comms_test_slave.py  —  run this on the SLAVE (servo) Pi

Listens on port 5000 for messages from the master Pi.
Prints each message it receives and echoes it back.

Usage:
    python3 comms_test_slave.py
"""

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/message", methods=["POST"])
def message():
    body = request.get_json(silent=True) or {}
    text = body.get("msg", "")
    print(f"[slave] received: {text!r}")
    reply = f"echo from slave: {text}"
    print(f"[slave] sending:  {reply!r}")
    return jsonify({"msg": reply, "from": "slave"})

@app.route("/ping")
def ping():
    return jsonify({"msg": "pong", "from": "slave"})

if __name__ == "__main__":
    print("Slave listening on port 5000 ...")
    app.run(host="0.0.0.0", port=5000)
