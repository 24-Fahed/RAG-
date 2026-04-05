"""推理服务 Worker 配置 - 从 deploy.yaml 读取配置。"""

from common.config import get_section, get_mode

_cfg = get_section("inference")

# --- 运行模式 ---
MODE = get_mode()

# --- 计算设备 ---
DEVICE = "cuda" if MODE != "mock" else "cpu"

# --- 模型路径 ---
LLM_MODEL_PATH = _cfg.get("llm_model_path", "/root/models/llama3-8b-ragga")
CLASSIFICATION_MODEL = _cfg.get("classification_model", "google-bert/bert-base-multilingual-cased")
EMBEDDING_MODEL = _cfg.get("embedding_model", "BAAI/bge-base-en-v1.5")
MONOT5_MODEL = _cfg.get("monot5_model", "castorini/monot5-base-msmarco-10k")
BGE_RERANKER_MODEL = _cfg.get("bge_reranker_model", "BAAI/bge-reranker-v2-m3")
RECOMP_EXTRACTIVE_MODEL = _cfg.get("recomp_extractive_model", "fangyuan/nq_extractive_compressor")
RECOMP_ABSTRACTIVE_MODEL = _cfg.get("recomp_abstractive_model", "fangyuan/nq_abstractive_compressor")
LLMLINGUA_MODEL = _cfg.get("llmlingua_model", "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank")

# --- 服务器配置 ---
HOST = _cfg.get("host", "0.0.0.0")
PORT = int(_cfg.get("port", 8000))
