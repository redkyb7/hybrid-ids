import json
import sqlite3
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_DIR.parent
DATABASE_PATH = PROJECT_ROOT / "data" / "ids_logs.db"
METRICS_PATH = PROJECT_ROOT / "deep learning model" / "saved_model" / "evaluation_metrics.json"
NORMAL_LABELS = {"Normal Traffic", "Normal"}

app = FastAPI(title="SentinelFlow API", version="1.0.0")
templates = Jinja2Templates(directory=str(FRONTEND_DIR / "templates"))


def load_model_metrics() -> dict[str, float]:
    if not METRICS_PATH.exists():
        return {}
    try:
        with METRICS_PATH.open(encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
        return {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def read_telemetry(limit: int) -> dict[str, Any]:
    if not DATABASE_PATH.exists():
        return {"error": "Telemetry database not found", "database_path": str(DATABASE_PATH)}

    try:
        with sqlite3.connect(DATABASE_PATH, timeout=5.0) as connection:
            connection.row_factory = sqlite3.Row
            columns = {row[1] for row in connection.execute("PRAGMA table_info(logs)")}
            for column, definition in {
                "source_port": "INTEGER",
                "destination_port": "INTEGER",
                "confidence": "REAL",
            }.items():
                if column not in columns:
                    connection.execute(f"ALTER TABLE logs ADD COLUMN {column} {definition}")
            logs = [
                dict(row)
                for row in connection.execute(
                    """
                          SELECT id, timestamp, source_ip, source_port,
                              destination_ip, destination_port, protocol,
                              attack_type, confidence, latency_ms
                    FROM logs
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            ]
            threat_rows = connection.execute(
                "SELECT attack_type, COUNT(*) AS count FROM logs GROUP BY attack_type"
            ).fetchall()

        threat_mix = {row["attack_type"]: row["count"] for row in threat_rows}
        attacks = [log for log in logs if log["attack_type"] not in NORMAL_LABELS]
        latencies = [log["latency_ms"] for log in logs if log["latency_ms"] is not None]
        metrics = load_model_metrics()
        current_f1 = metrics.get("macro_f1")
        total_flows = sum(threat_mix.values())
        average_latency = round(sum(latencies) / len(latencies)) if latencies else 0

        return {
            "logs": logs,
            "kpis": {
                "avg_latency": average_latency,
                "current_f1": current_f1,
                "f1_target": 0.70,
                "f1_improvement": round(((current_f1 - 0.70) / 0.70) * 100, 1)
                if current_f1 is not None
                else None,
                "total_attacks": sum(
                    count for label, count in threat_mix.items() if label not in NORMAL_LABELS
                ),
                "total_flows": total_flows,
                "window_attacks": len(attacks),
                "window_size": len(logs),
                "model_accuracy": metrics.get("accuracy"),
            },
            "threat_mix": threat_mix,
            "latest_alert": attacks[0] if attacks else None,
            "database_path": str(DATABASE_PATH),
        }
    except sqlite3.Error as error:
        return {"error": f"Could not read telemetry database: {error}"}


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get("/api/telemetry")
async def get_telemetry(
    limit: int = Query(default=500, ge=1, le=2500),
) -> dict[str, Any]:
    return read_telemetry(limit)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "database": "ready" if DATABASE_PATH.exists() else "waiting"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
