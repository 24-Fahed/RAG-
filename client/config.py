"""客户端配置 - 从自身 config/ 目录读取。"""

import yaml
from pathlib import Path

_CONFIG_DIR = Path(__file__).parent / "config"

# 以下变量在 load_config() 调用后才有值
RAG_SERVER_URL = ""


def load_config(mode: str):
    """加载指定模式的配置文件。必须在导入其他模块之前调用。"""
    global RAG_SERVER_URL

    path = _CONFIG_DIR / f"{mode}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    client_cfg = cfg.get("client", {})
    RAG_SERVER_URL = client_cfg.get("server_url", "http://127.0.0.1:8000")
