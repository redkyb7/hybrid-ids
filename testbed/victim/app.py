"""
SentinelFlow Testbed - Victim Web Application & Vulnerable Service
==================================================================
Simulates realistic enterprise web services including:
  1. Benign corporate home page & API status endpoints
  2. Authentication endpoint (/login) susceptible to credential brute-forcing
  3. Search / Query endpoint (/search) susceptible to SQL Injection / XSS
"""

from flask import Flask, Response, request, jsonify, render_template_string
import hashlib
import secrets
import socket
import sqlite3
import os
import threading
import time

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8192

# In-memory mock database for SQL injection testing
def init_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    c = conn.cursor()
    c.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, role TEXT)")
    c.execute("INSERT INTO users VALUES (1, 'admin', 'Administrator')")
    c.execute("INSERT INTO users VALUES (2, 'user1', 'Developer')")
    c.execute("INSERT INTO users VALUES (3, 'user2', 'Analyst')")
    conn.commit()
    return conn

mock_db = init_db()
_mock_db_lock = threading.Lock()

# Synthetic testbed fixtures. These values are generated in memory and contain
# no user data. Tokens only authorize access to the lab canary endpoints.
LAB_CANARY_PATTERN = b"SENTINELFLOW-LAB-CANARY-ONLY-"
_lab_tokens = {}
_lab_tokens_lock = threading.Lock()
_udp_stats = {"datagrams": 0, "bytes": 0}
_udp_stats_lock = threading.Lock()


def _bounded_int(name, default, minimum, maximum):
    try:
        value = int(request.args.get(name, default))
    except (TypeError, ValueError):
        return None
    return value if minimum <= value <= maximum else None


def _lab_authorized():
    authorization = request.headers.get("Authorization", "")
    if not authorization.startswith("Bearer "):
        return False
    token = authorization[7:]
    with _lab_tokens_lock:
        expires = _lab_tokens.get(token, 0)
        if expires <= time.monotonic():
            _lab_tokens.pop(token, None)
            return False
        return True


def _canary(size):
    return (LAB_CANARY_PATTERN * ((size // len(LAB_CANARY_PATTERN)) + 1))[:size]


def run_udp_sink():
    """Receive small one-way UDP lab packets; never reply or amplify them."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as listener:
        listener.bind(("0.0.0.0", 9999))
        while True:
            payload, _ = listener.recvfrom(1024)
            with _udp_stats_lock:
                _udp_stats["datagrams"] += 1
                _udp_stats["bytes"] += len(payload)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Enterprise Corporate Portal</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f8fafc; color: #1e293b; }
        .card { background: white; padding: 24px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); max-width: 600px; margin-bottom: 20px; }
        input[type=text], input[type=password] { width: 100%; padding: 8px 12px; margin: 8px 0; border: 1px solid #cbd5e1; border-radius: 4px; box-sizing: border-box; }
        button { background: #0284c7; color: white; border: none; padding: 10px 16px; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0369a1; }
        .badge { display: inline-block; background: #e0f2fe; color: #0369a1; padding: 4px 8px; border-radius: 4px; font-size: 0.85em; }
    </style>
</head>
<body>
    <div class="card">
        <h2>🛡️ Enterprise Portal Target Node (Victim)</h2>
        <p>IP Address: <span class="badge">192.168.100.10</span></p>
        <p>This service provides normal and vulnerable endpoints for Hybrid IDS testbed evaluation.</p>
    </div>

    <div class="card">
        <h3>Employee Directory Search (SQLi Endpoint)</h3>
        <form method="GET" action="/search">
            <input type="text" name="q" placeholder="Search by username (e.g. admin or ' OR 1=1--)" value="{{ query|default('') }}">
            <button type="submit">Search</button>
        </form>
        {% if results %}
            <div style="margin-top: 15px;">
                <strong>Results:</strong>
                <ul>
                {% for r in results %}
                    <li>{{ r[1] }} — <em>{{ r[2] }}</em></li>
                {% endfor %}
                </ul>
            </div>
        {% endif %}
    </div>

    <div class="card">
        <h3>User Authentication (/login)</h3>
        <form method="POST" action="/login">
            <input type="text" name="username" placeholder="Username">
            <input type="password" name="password" placeholder="Password">
            <button type="submit">Authenticate</button>
        </form>
        {% if msg %}<p style="color: red; margin-top: 10px;">{{ msg }}</p>{% endif %}
    </div>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/status")
def status():
    """Benign API endpoint called by legitimate client."""
    return jsonify({
        "status": "healthy",
        "service": "corporate-portal",
        "uptime": "99.98%",
        "node_ip": request.host
    })

@app.route("/api/data")
def data():
    """Benign data payload endpoint."""
    return jsonify({
        "metrics": [10, 25, 40, 15, 30],
        "message": "Authenticated flow telemetry normal"
    })


@app.route("/lab/load")
def lab_load():
    """Bounded response size and delay for HTTP flow-profile experiments."""
    size = _bounded_int("size", 935, 0, 2048)
    delay_ms = _bounded_int("delay_ms", 0, 0, 2000)
    if size is None or delay_ms is None:
        return jsonify({"error": "lab load limits exceeded"}), 400
    time.sleep(delay_ms / 1000)
    return Response(b"L" * size, mimetype="application/octet-stream")


@app.route("/lab/c2")
def lab_c2():
    """Small, deterministic reply for the botnet-checkin lab scenario."""
    size = _bounded_int("size", 129, 0, 256)
    delay_ms = _bounded_int("delay_ms", 10, 0, 500)
    if size is None or delay_ms is None:
        return jsonify({"error": "lab C2 limits exceeded"}), 400
    time.sleep(delay_ms / 1000)
    return Response(b"C" * size, mimetype="application/octet-stream")


@app.route("/lab/auth", methods=["POST"])
def lab_auth():
    """Issue a short-lived token only for synthetic canary testing."""
    if (request.form.get("username") != "lab"
            or request.form.get("password") != "lab-only-password"):
        return jsonify({"error": "invalid lab credentials"}), 401
    token = secrets.token_urlsafe(24)
    with _lab_tokens_lock:
        _lab_tokens[token] = time.monotonic() + 60
    return jsonify({"token": token, "expires_seconds": 60})


@app.route("/lab/canary")
def lab_canary():
    if not _lab_authorized():
        return jsonify({"error": "lab authorization required"}), 401
    size = _bounded_int("size", 1024, 1, 4096)
    if size is None:
        return jsonify({"error": "lab canary size exceeded"}), 400
    return Response(_canary(size), mimetype="application/octet-stream")


@app.route("/lab/collect", methods=["POST"])
def lab_collect():
    if not _lab_authorized():
        return jsonify({"error": "lab authorization required"}), 401
    if request.content_length is not None and request.content_length > 4096:
        return jsonify({"error": "lab collection size exceeded"}), 413
    content = request.get_data(cache=False)
    if not 1 <= len(content) <= 4096:
        return jsonify({"error": "lab collection size exceeded"}), 413
    if content != _canary(len(content)):
        return jsonify({"error": "only synthetic lab canary accepted"}), 400
    return jsonify({"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})


@app.route("/lab/udp-stats")
def lab_udp_stats():
    with _udp_stats_lock:
        return jsonify(dict(_udp_stats))

@app.route("/search")
def search():
    """Vulnerable search endpoint simulating Web Attack (SQL Injection / XSS)."""
    q = request.args.get("q", "")
    results = []
    if q:
        # Deliberate vulnerable string interpolation for IDS attack simulation
        try:
            with _mock_db_lock:
                c = mock_db.cursor()
                query = f"SELECT id, username, role FROM users WHERE username = '{q}'"
                c.execute(query)
                results = c.fetchall()
        except Exception:
            results = [(1, "SQL_ERROR", "Syntax exception")]
    return render_template_string(HTML_TEMPLATE, query=q, results=results)

@app.route("/login", methods=["POST"])
def login():
    """Vulnerable endpoint simulating authentication brute force."""
    user = request.form.get("username", "")
    pwd = request.form.get("password", "")
    if user == "admin" and pwd == "password123":
        return render_template_string(HTML_TEMPLATE, msg="Authentication Successful! (Admin Access Granted)")
    return render_template_string(HTML_TEMPLATE, msg="Authentication Failed: Invalid Credentials"), 401


if __name__ == "__main__":
    threading.Thread(target=run_udp_sink, name="lab-udp-sink", daemon=True).start()
    app.run(host="0.0.0.0", port=80, threaded=True)
