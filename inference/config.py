"""推理服务 Worker 配置 - 从自身 config/ 目录读取。"""

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


def get_mode() -> str:
    return _load().get("mode", "mock")


# --- 延迟加载配置 ---
_cfg = _load()

# --- 运行模式 ---
MODE = get_mode()

# --- 计算设备 ---
DEVICE = "cuda" if MODE != "mock" else "cpu"

# --- 模型路径 ---
inference_cfg = _cfg.get("inference", {})
LLM_MODEL_PATH = inference_cfg.get("llm_model_path", "/root/models/llama3-8b-ragga")
CLASSIFICATION_MODEL = inference_cfg.get("classification_model", "google-bert/bert-base-multilingual-cased")
EMBEDDING_MODEL = inference_cfg.get("embedding_model", "BAAI/bge-base-en-v1.5")
MONOT5_MODEL = inference_cfg.get("monot5_model", "castorini/monot5-base-msmarco-10k")
BGE_RERANKER_MODEL = inference_cfg.get("bge_reranker_model", "BAAI/bge-reranker-v2-m3")
RECOMP_EXTRACTIVE_MODEL = inference_cfg.get("recomp_extractive_model", "fangyuan/nq_extractive_compressor")
RECOMP_ABSTRACTIVE_MODEL = inference_cfg.get("recomp_abstractive_model", "fangyuan/nq_abstractive_compressor")
LLMLINGUA_MODEL = inference_cfg.get("llmlingua_model", "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank")

# --- 服务器配置 ---
HOST = inference_cfg.get("host", "0.0.0.0")
PORT = int(inference_cfg.get("port", 8000))
