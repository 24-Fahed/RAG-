"""客户端配置 - 从自身 config/ 目录读取。"""

import yaml
from pathlib import Path

_CONFIG_DIR = Path(__file__).parent / "config"
_config_cache: dict | None = None
_current_mode: str = "local"


def load_config(mode: str):
    """加载指定模式的配置文件。必须在导入其他模块之前调用。"""
    global _config_cache, _current_mode
    _current_mode = mode
    _config_cache = None


def _load() -> dict:
    global _config_cache
    if _config_cache is None:
        path = _CONFIG_DIR / f"{_current_mode}.yaml"
        with open(path, "r", encoding="utf-8") as f:
            _config_cache = yaml.safe_load(f)
    return _config_cache


# --- 延迟加载配置 ---
_cfg = _load()
client_cfg = _cfg.get("client", {})
RAG_SERVER_URL = client_cfg.get("server_url", "http://127.0.0.1:8000")
