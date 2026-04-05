"""推理服务 Worker 配置 - 从自身 config/ 目录读取。"""

import yaml
from pathlib import Path

_CONFIG_DIR = Path(__file__).parent / "config"
_loaded = False

# 以下变量在 load_config() 调用后才有值
MODE = ""
DEVICE = ""
LLM_MODEL_PATH = ""
CLASSIFICATION_MODEL = ""
EMBEDDING_MODEL = ""
MONOT5_MODEL = ""
BGE_RERANKER_MODEL = ""
RECOMP_EXTRACTIVE_MODEL = ""
RECOMP_ABSTRACTIVE_MODEL = ""
LLMLINGUA_MODEL = ""
HOST = ""
PORT = 0


def load_config(mode: str):
    """加载指定模式的配置文件。必须在导入其他模块之前调用。"""
    global _loaded
    global MODE, DEVICE, LLM_MODEL_PATH, CLASSIFICATION_MODEL, EMBEDDING_MODEL
    global MONOT5_MODEL, BGE_RERANKER_MODEL, RECOMP_EXTRACTIVE_MODEL
    global RECOMP_ABSTRACTIVE_MODEL, LLMLINGUA_MODEL, HOST, PORT

    path = _CONFIG_DIR / f"{mode}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    MODE = cfg.get("mode")
    DEVICE = "cuda" if MODE != "mock" else "cpu"
    inf = cfg.get("inference", {})
    LLM_MODEL_PATH = inf.get("llm_model_path", "")
    CLASSIFICATION_MODEL = inf.get("classification_model", "")
    EMBEDDING_MODEL = inf.get("embedding_model", "")
    MONOT5_MODEL = inf.get("monot5_model", "")
    BGE_RERANKER_MODEL = inf.get("bge_reranker_model", "")
    RECOMP_EXTRACTIVE_MODEL = inf.get("recomp_extractive_model", "")
    RECOMP_ABSTRACTIVE_MODEL = inf.get("recomp_abstractive_model", "")
    LLMLINGUA_MODEL = inf.get("llmlingua_model", "")
    HOST = inf.get("host", "0.0.0.0")
    PORT = int(inf.get("port", 8000))
    _loaded = True
