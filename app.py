"""Production WSGI entrypoint: gunicorn app:server --bind 0.0.0.0:8050."""

import os
from pathlib import Path

from mobility.dashboard.app import create_app
from mobility.pipeline import current_snapshot

project_root = Path(__file__).resolve().parent
snapshot_path = os.environ.get("MOBILITY_SNAPSHOT")
snapshot = Path(snapshot_path) if snapshot_path else current_snapshot(project_root / "artifacts")
app = create_app(snapshot)
server = app.server

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8050")), debug=False)
