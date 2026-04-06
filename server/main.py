"""RAG server FastAPI entrypoint."""

from __future__ import annotations

import argparse
import logging
import os

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--mode",
    default="local",
    choices=["local", "staging", "production"],
    help="Deployment mode: local / staging / production (default: local)",
)
_args, _ = _parser.parse_known_args()

from server.config import load_config

load_config(_args.mode)

from fastapi import FastAPI

from runtime_diagnostics import install_runtime_diagnostics
from server.config import HOST, PORT
from server.routers import indexing, query

app = FastAPI(title="RAG Server", version="0.1.0")
install_runtime_diagnostics(app, service_name="server", mode=_args.mode)

app.include_router(query.router, prefix="/api", tags=["query"])
app.include_router(indexing.router, prefix="/api", tags=["indexing"])


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    reload = False
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    print(f"RAG Server starting at {HOST}:{PORT}")
    print(f"Config file: config/{_args.mode}.yaml")
    print(f"PID={os.getpid()} PPID={os.getppid()} reload={reload}")
    uvicorn.run("server.main:app", host=HOST, port=PORT, reload=reload)
