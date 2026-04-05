"""客户端配置 - 从 deploy.yaml 中读取。"""

from common.config import get_section

_cfg = get_section("client")
RAG_SERVER_URL = _cfg.get("server_url", "http://127.0.0.1:8000")
