"""服务端配置 - 从自身 config/ 目录读取。"""

import yaml
from pathlib import Path

_CONFIG_DIR = Path(__file__).parent / "config"

# 以下变量在 load_config() 调用后才有值
MILVUS_HOST = ""
MILVUS_PORT = 0
INFERENCE_URL = ""
BM25_INDEX_DIR = ""
COLLECTION_NAME = ""
EMBEDDING_DIM = 0
HOST = ""
PORT = 0
CHUNK_SIZE = 512
CHUNK_OVERLAP = 20

DEFAULT_SEARCH_METHOD = "hyde_with_hybrid"
DEFAULT_RERANK_MODEL = "monot5"
DEFAULT_TOP_K = 10
DEFAULT_COMPRESSION_RATIO = 0.6
DEFAULT_REPACK_METHOD = "sides"
DEFAULT_HYBRID_ALPHA = 0.3
DEFAULT_SEARCH_K = 100


def load_config(mode: str):
    """加载指定模式的配置文件。必须在导入其他模块之前调用。"""
    global MILVUS_HOST, MILVUS_PORT, INFERENCE_URL, BM25_INDEX_DIR
    global COLLECTION_NAME, EMBEDDING_DIM, HOST, PORT

    path = _CONFIG_DIR / f"{mode}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    srv = cfg.get("server", {})
    MILVUS_HOST = srv.get("milvus_host", "localhost")
    MILVUS_PORT = int(srv.get("milvus_port", 19530))
    INFERENCE_URL = srv.get("inference_url", "http://localhost:8001")
    BM25_INDEX_DIR = srv.get("bm25_index_dir", "./bm25_data")
    COLLECTION_NAME = srv.get("collection_name", "rag_collection")
    EMBEDDING_DIM = int(srv.get("embedding_dim", 768))
    HOST = srv.get("host", "0.0.0.0")
    PORT = int(srv.get("port", 8000))
