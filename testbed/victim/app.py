"""
SentinelFlow Testbed - Victim Web Application & Vulnerable Service
==================================================================
Simulates realistic enterprise web services including:
  1. Benign corporate home page & API status endpoints
  2. Authentication endpoint (/login) susceptible to credential brute-forcing
  3. Search / Query endpoint (/search) susceptible to SQL Injection / XSS
"""

from flask import Flask, request, jsonify, render_template_string
import sqlite3
import os

app = Flask(__name__)

# In-memory mock database for SQL injection testing
def init_db():
    conn = sqlite3.connect(":memory:")
    c = conn.cursor()
    c.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, role TEXT)")
    c.execute("INSERT INTO users VALUES (1, 'admin', 'Administrator')")
    c.execute("INSERT INTO users VALUES (2, 'user1', 'Developer')")
    c.execute("INSERT INTO users VALUES (3, 'user2', 'Analyst')")
    conn.commit()
    return conn

mock_db = init_db()

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

@app.route("/search")
def search():
    """Vulnerable search endpoint simulating Web Attack (SQL Injection / XSS)."""
    q = request.args.get("q", "")
    results = []
    if q:
        # Deliberate vulnerable string interpolation for IDS attack simulation
        try:
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
    app.run(host="0.0.0.0", port=80)
