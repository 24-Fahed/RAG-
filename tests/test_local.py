"""Local end-to-end smoke test for the retrieval-oriented RAG knowledge base."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "local_test_data"
RAG_SERVER = "http://127.0.0.1:8001"
INFERENCE_SERVER = "http://127.0.0.1:8000"
COLLECTION = "rag_collection"

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
        return

    failed += 1
    msg = f"  FAIL  {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)


def test_health() -> bool:
    print("\n[Step 1] Health")

    try:
        inference = httpx.get(f"{INFERENCE_SERVER}/health", timeout=5).json()
        check("Inference worker online", inference.get("status") == "ok")
        check("Inference worker in mock mode", inference.get("mode") == "mock", str(inference))
    except Exception as exc:
        check("Inference worker online", False, str(exc))
        return False

    try:
        rag = httpx.get(f"{RAG_SERVER}/health", timeout=5).json()
        check("RAG server online", rag.get("status") == "ok")
    except Exception as exc:
        check("RAG server online", False, str(exc))
        return False

    return True


def test_inference_endpoints() -> None:
    print("\n[Step 2] Inference endpoints")

    try:
        data = httpx.post(f"{INFERENCE_SERVER}/inference/classify", json={"query": "What is RAG?"}, timeout=10).json()
        check("classify returns label", "label" in data, str(data))
    except Exception as exc:
        check("classify returns label", False, str(exc))

    try:
        data = httpx.post(f"{INFERENCE_SERVER}/inference/embed", json={"texts": ["test text"]}, timeout=10).json()
        dim = len(data.get("embeddings", [[]])[0])
        check("embed returns 768 dimensions", dim == 768, str(dim))
    except Exception as exc:
        check("embed returns 768 dimensions", False, str(exc))

    try:
        data = httpx.post(f"{INFERENCE_SERVER}/inference/hyde", json={"query": "What is RAG?"}, timeout=10).json()
        check("hyde returns hypothetical_document", bool(data.get("hypothetical_document")))
    except Exception as exc:
        check("hyde returns hypothetical_document", False, str(exc))

    try:
        data = httpx.post(
            f"{INFERENCE_SERVER}/inference/rerank",
            json={
                "query": "What is RAG?",
                "documents": [{"content": "doc1", "score": 0.5, "metadata": {}}],
                "model": "monot5",
                "top_k": 5,
            },
            timeout=10,
        ).json()
        check("rerank returns documents", len(data.get("documents", [])) > 0, str(data))
    except Exception as exc:
        check("rerank returns documents", False, str(exc))

    try:
        data = httpx.post(
            f"{INFERENCE_SERVER}/inference/compress",
            json={
                "query": "What is RAG?",
                "context": "RAG retrieves relevant passages and compresses context for downstream use.",
                "method": "recomp_extractive",
                "ratio": 0.6,
            },
            timeout=10,
        ).json()
        check("compress returns text", bool(data.get("compressed")))
    except Exception as exc:
        check("compress returns text", False, str(exc))


def test_index() -> bool:
    print("\n[Step 3] Index")

    if not TEST_DATA_DIR.exists():
        check("Test data directory exists", False, str(TEST_DATA_DIR))
        return False

    files = [open(path, "rb") for path in sorted(TEST_DATA_DIR.iterdir()) if path.is_file()]
    check("Found test files", len(files) > 0, str(len(files)))
    if not files:
        return False

    try:
        resp = httpx.post(
            f"{RAG_SERVER}/api/index",
            files=[("files", fh) for fh in files],
            data={"collection": COLLECTION},
            timeout=300,
        )
        data = resp.json()
        check("Index request succeeded", resp.status_code == 200, resp.text[:200])
        check("Index response status ok", data.get("status") == "ok", str(data))
        check("Indexed chunks > 0", data.get("document_count", 0) > 0, str(data))
        return resp.status_code == 200 and data.get("status") == "ok"
    except Exception as exc:
        check("Index request succeeded", False, str(exc))
        return False
    finally:
        for fh in files:
            fh.close()


def test_query() -> None:
    print("\n[Step 4] Query")

    queries = [
        "What is RAG?",
        "How does Milvus work?",
        "What models does the system use?",
    ]

    for query in queries:
        print(f"\n  Query: {query}")
        try:
            resp = httpx.post(
                f"{RAG_SERVER}/api/query",
                json={"query": query, "top_k": 5},
                timeout=120,
            )
            data = resp.json()

            retrieved = data.get("retrieved_documents", [])
            reranked = data.get("reranked_documents", [])
            repacked_context = data.get("repacked_context", "")
            compressed_context = data.get("compressed_context", "")

            print(f"  classification={data.get('classification_label')}")
            print(f"  retrieved={len(retrieved)} reranked={len(reranked)}")
            print(f"  repacked_chars={len(repacked_context)} compressed_chars={len(compressed_context)}")

            check("Query request succeeded", resp.status_code == 200, resp.text[:200])
            check("HyDE returned", bool(data.get("hyde_document")))
            check("Retrieved documents returned", len(retrieved) > 0, str(len(retrieved)))
            check("Reranked documents returned", len(reranked) > 0, str(len(reranked)))
            check("Repacked context returned", bool(repacked_context))
            check("Compressed context returned", bool(compressed_context))
        except Exception as exc:
            check("Query request succeeded", False, str(exc))


def test_cleanup() -> None:
    print("\n[Step 5] Cleanup")

    try:
        from pymilvus import MilvusClient

        client = MilvusClient(uri="http://localhost:19530")
        if client.has_collection(COLLECTION):
            client.drop_collection(COLLECTION)
        check("Milvus collection cleaned", True)
    except Exception as exc:
        check("Milvus collection cleaned", False, str(exc))

    bm25_file = PROJECT_ROOT / "bm25_data" / f"{COLLECTION}.pkl"
    try:
        if bm25_file.exists():
            os.remove(bm25_file)
        check("BM25 file cleaned", True)
    except Exception as exc:
        check("BM25 file cleaned", False, str(exc))


def main() -> None:
    print("=" * 60)
    print("RAG local smoke test")
    print("=" * 60)

    if not test_health():
        print("\nServices are not ready.")
        sys.exit(1)

    try:
        test_inference_endpoints()
        if test_index():
            test_query()
    finally:
        test_cleanup()

    print("\n" + "=" * 60)
    print(f"Test result: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
