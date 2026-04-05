"""服务端配置 - 从 deploy.yaml 读取。"""

from common.config import get_section, is_mock

_cfg = get_section("server")

# --- Milvus 配置 ---
MILVUS_HOST = _cfg.get("milvus_host", "localhost")
MILVUS_PORT = int(_cfg.get("milvus_port", 19530))

# --- 推理 Worker 配置 ---
INFERENCE_URL = _cfg.get("inference_url", "http://localhost:8001")

# --- BM25 配置 ---
BM25_INDEX_DIR = _cfg.get("bm25_index_dir", "./bm25_data")

# --- Collection 配置 ---
COLLECTION_NAME = _cfg.get("collection_name", "rag_collection")
EMBEDDING_DIM = int(_cfg.get("embedding_dim", 768))

# --- 服务端配置 ---
HOST = _cfg.get("host", "0.0.0.0")
PORT = int(_cfg.get("port", 8000))

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
