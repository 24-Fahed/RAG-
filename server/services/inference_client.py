"""用于调用 GPU 服务器上推理 Worker 的 HTTP 客户端。

RAG 服务端通过 HTTP 与推理 Worker 通信。
当 GPU 服务器在 AutoDL 上（无公网 IP）时，RAG 服务端
通过 SSH 隧道访问它：

    ssh -N -L 8001:localhost:8000 -p <port> root@connect.westb.seetacloud.com

推理 Worker 随后可通过 http://localhost:8001 访问。
"""

import httpx
from server.config import INFERENCE_URL

_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.Client(base_url=INFERENCE_URL, timeout=300.0)
    return _client


# ---------- 分类 ----------

def classify(query: str) -> int:
    resp = _get_client().post("/inference/classify", json={"query": query})
    resp.raise_for_status()
    return resp.json()["label"]


# ---------- 嵌入 ----------

def embed(texts: list[str]) -> list[list[float]]:
    resp = _get_client().post("/inference/embed", json={"texts": texts})
    resp.raise_for_status()
    data = resp.json()
    return data["embeddings"]


# ---------- 重排序 ----------

def rerank(query: str, documents: list[dict], model: str = "monot5", top_k: int = 10) -> list[dict]:
    resp = _get_client().post("/inference/rerank", json={
        "query": query,
        "documents": documents,
        "model": model,
        "top_k": top_k,
    })
    resp.raise_for_status()
    return resp.json()["documents"]


# ---------- 压缩 ----------

def compress(query: str, context: str, method: str = "recomp_extractive", ratio: float = 0.6) -> str:
    resp = _get_client().post("/inference/compress", json={
        "query": query,
        "context": context,
        "method": method,
        "ratio": ratio,
    })
    resp.raise_for_status()
    return resp.json()["compressed"]


# ---------- 生成 ----------

def generate(query: str, context: str = "", max_out_len: int = 50) -> str:
    resp = _get_client().post("/inference/generate", json={
        "query": query,
        "context": context,
        "max_out_len": max_out_len,
    })
    resp.raise_for_status()
    return resp.json()["answer"]


def hyde(query: str, max_out_len: int = 100) -> str:
    resp = _get_client().post("/inference/hyde", json={
        "query": query,
        "max_out_len": max_out_len,
    })
    resp.raise_for_status()
    return resp.json()["hypothetical_document"]
