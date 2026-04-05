"""服务端配置 - 从自身 config/ 目录读取。"""

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
server_cfg = _cfg.get("server", {})

# --- Milvus 配置 ---
MILVUS_HOST = server_cfg.get("milvus_host", "localhost")
MILVUS_PORT = int(server_cfg.get("milvus_port", 19530))

# --- 推理 Worker 配置 ---
INFERENCE_URL = server_cfg.get("inference_url", "http://localhost:8001")

# --- BM25 配置 ---
BM25_INDEX_DIR = server_cfg.get("bm25_index_dir", "./bm25_data")

# --- Collection 配置 ---
COLLECTION_NAME = server_cfg.get("collection_name", "rag_collection")
EMBEDDING_DIM = int(server_cfg.get("embedding_dim", 768))

# --- 服务端配置 ---
HOST = server_cfg.get("host", "0.0.0.0")
PORT = int(server_cfg.get("port", 8000))

# --- 流水线默认参数 ---
DEFAULT_SEARCH_METHOD = "hyde_with_hybrid"
DEFAULT_RERANK_MODEL = "monot5"
DEFAULT_TOP_K = 10
DEFAULT_COMPRESSION_RATIO = 0.6
DEFAULT_REPACK_METHOD = "sides"
DEFAULT_HYBRID_ALPHA = 0.3
DEFAULT_SEARCH_K = 100

# --- 索引配置 ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 20
