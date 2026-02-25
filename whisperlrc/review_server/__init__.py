from __future__ import annotations

from pathlib import Path


def run_review_server(
    output_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    config_path: Path | None = None,
) -> None:
    from uvicorn import Config, Server

    from whisperlrc.review_server.app import create_app

    app = create_app(output_dir=output_dir, config_path=config_path)
    cfg = Config(app=app, host=host, port=port, log_level="info")
    server = Server(cfg)
    server.run()
