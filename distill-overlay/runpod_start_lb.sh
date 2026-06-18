#!/usr/bin/env bash
set -euo pipefail

cd /app

exec uv run python - <<'PY'
import os

import uvicorn
from fastapi import Response

import api_server


def ping() -> Response:
    return Response(status_code=200)


api_server.app.add_api_route("/ping", ping, methods=["GET"])
uvicorn.run(
    api_server.app,
    host="0.0.0.0",
    port=int(os.getenv("PORT", "8000")),
)
PY
