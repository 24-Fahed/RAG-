"""Staging end-to-end smoke test for the deployed RAG service.

This script focuses on operational confidence instead of strict retrieval
benchmarking:

1. Check the public server health endpoint.
2. Download a small SciFact subset from Hugging Face.
3. Upload and index a capped document set through `/api/index`.
4. Run a capped query set through `/api/query`.
5. Print actionable diagnostics when the server returns non-200 responses.
6. Clean up the temporary Milvus collection and local temp files.

Usage:
    python tests/test_staging.py
    python tests/test_staging.py --server-url http://<server>:8000
    python tests/test_staging.py --server-url http://<server>:8000 --max-docs 500 --max-queries 5
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_COLLECTION = "scifact_test"
TEMP_DIR = PROJECT_ROOT / "tests" / "staging_test_data"

HEALTH_TIMEOUT = 20
INDEX_TIMEOUT = 1200
QUERY_TIMEOUT = 240

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default=None, help="Override RAG server URL")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Milvus collection for this test run")
    parser.add_argument("--max-docs", type=int, default=1000, help="Maximum number of corpus docs to upload")
    parser.add_argument("--max-queries", type=int, default=10, help="Maximum number of SciFact queries to test")
    return parser.parse_args()


def get_server_url(cli_url: str | None) -> str:
    if cli_url:
        return cli_url.rstrip("/")

    try:
        from client.config import load_config

        load_config("staging")
        from client.config import RAG_SERVER_URL

        return RAG_SERVER_URL.rstrip("/")
    except Exception:
        return "http://127.0.0.1:8001"


def safe_preview(text: str, limit: int = 300) -> str:
    text = text.replace("\n", "\\n")
    return text[:limit]


def print_documents(title: str, documents: list[dict], *, limit: int = 10) -> None:
    print(f"    {title} ({len(documents)}):")
    if not documents:
        print("      <empty>")
        return

    for idx, doc in enumerate(documents[:limit], start=1):
        score = doc.get("score")
        content = safe_preview(doc.get("content", ""), 240)
        metadata = doc.get("metadata", {})
        print(f"      [{idx}] score={score} metadata={metadata}")
        print(f"          content={content}")


def http_error_detail(resp: httpx.Response) -> str:
    body = safe_preview(resp.text)
    return f"HTTP {resp.status_code} | body={body}"


def request_json(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    timeout: float,
    expect_status: int = 200,
    **kwargs,
):
    start = time.time()
    resp = client.request(method, path, timeout=timeout, **kwargs)
    elapsed = time.time() - start
    if resp.status_code != expect_status:
        raise RuntimeError(f"{method} {path} failed after {elapsed:.1f}s: {http_error_detail(resp)}")
    return resp.json(), elapsed


def test_health(client: httpx.Client) -> bool:
    print("\n[Step 1] Server health")
    try:
        data, elapsed = request_json(client, "GET", "/health", timeout=HEALTH_TIMEOUT)
        check("RAG Server online", data.get("status") == "ok", f"elapsed={elapsed:.1f}s")
        return data.get("status") == "ok"
    except Exception as exc:
        check("RAG Server online", False, str(exc))
        return False


def download_scifact():
    print("\n[Step 2] Download SciFact")
    try:
        from datasets import load_dataset
    except ImportError:
        check("datasets installed", False, "pip install datasets")
        return None, None, None

    try:
        corpus_ds = load_dataset("BeIR/scifact", "corpus", split="corpus", trust_remote_code=True)
        check("Downloaded corpus", True, f"{len(corpus_ds)} docs")
    except Exception as exc:
        check("Downloaded corpus", False, str(exc))
        return None, None, None

    try:
        queries_ds = load_dataset("BeIR/scifact", "queries", split="queries", trust_remote_code=True)
        check("Downloaded queries", True, f"{len(queries_ds)} queries")
    except Exception as exc:
        check("Downloaded queries", False, str(exc))
        return corpus_ds, None, None

    try:
        qrels_ds = load_dataset("BeIR/scifact-qrels", split="test", trust_remote_code=True)
        check("Downloaded qrels", True, f"{len(qrels_ds)} labels")
    except Exception as exc:
        check("Downloaded qrels", False, str(exc))
        return corpus_ds, queries_ds, None

    return corpus_ds, queries_ds, qrels_ds


def build_temp_corpus(corpus_ds, max_docs: int) -> int:
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for item in corpus_ds:
        doc_id = item.get("_id", count)
        title = item.get("title", "")
        text = item.get("text", "")
        content = f"{title}\n\n{text}" if title else text
        filepath = TEMP_DIR / f"doc_{doc_id}.txt"
        filepath.write_text(content, encoding="utf-8")
        count += 1
        if count >= max_docs:
            break

    check("Prepared temporary corpus files", count > 0, f"{count} files")
    return count


def test_index(client: httpx.Client, corpus_ds, collection: str, max_docs: int) -> bool:
    print("\n[Step 3] Index SciFact subset")
    file_handles = []

    try:
        file_count = build_temp_corpus(corpus_ds, max_docs)
        if file_count == 0:
            return False

        for fname in sorted(os.listdir(TEMP_DIR)):
            fpath = TEMP_DIR / fname
            if fpath.is_file() and fpath.suffix == ".txt":
                file_handles.append(open(fpath, "rb"))

        start = time.time()
        resp = client.post(
            "/api/index",
            files=[("files", fh) for fh in file_handles],
            data={"collection": collection},
            timeout=INDEX_TIMEOUT,
        )
        elapsed = time.time() - start

        if resp.status_code != 200:
            check("Index request succeeded", False, http_error_detail(resp))
            return False

        data = resp.json()
        check("Index response status ok", data.get("status") == "ok", f"elapsed={elapsed:.1f}s")
        check("Indexed documents > 0", data.get("document_count", 0) > 0, str(data.get("document_count")))
        print(f"    Indexed {data.get('document_count', 0)} chunks in {elapsed:.1f}s")
        return data.get("status") == "ok"
    except Exception as exc:
        check("Index request succeeded", False, str(exc))
        return False
    finally:
        for fh in file_handles:
            fh.close()


def build_relevance_lookup(qrels_ds, corpus_ds):
    qrel_map: dict[str, set[str]] = {}
    title_map: dict[str, str] = {}

    if qrels_ds is not None:
        for item in qrels_ds:
            qid = str(item.get("query-id", ""))
            did = str(item.get("corpus-id", ""))
            qrel_map.setdefault(qid, set()).add(did)

    if corpus_ds is not None:
        for item in corpus_ds:
            doc_id = str(item.get("_id", ""))
            title_map[doc_id] = item.get("title", "") or ""

    return qrel_map, title_map


def query_payload(query: str, collection: str) -> dict:
    return {
        "query": query,
        "collection": collection,
        "top_k": 10,
    }


def test_queries(
    client: httpx.Client,
    queries_ds,
    qrels_ds,
    corpus_ds,
    max_queries: int,
    collection: str,
) -> None:
    print("\n[Step 4] Query smoke test")

    test_queries_list = []
    if queries_ds is not None:
        for item in queries_ds:
            test_queries_list.append(
                {
                    "id": str(item.get("_id", "")),
                    "text": item.get("text", ""),
                }
            )
    test_queries_list = test_queries_list[:max_queries]
    check("Prepared test queries", len(test_queries_list) > 0, str(len(test_queries_list)))
    if not test_queries_list:
        return

    qrel_map, title_map = build_relevance_lookup(qrels_ds, corpus_ds)

    query_success = 0
    relevant_hit = 0
    query_with_qrels = 0

    for item in test_queries_list:
        qid = item["id"]
        qtext = item["text"]
        print(f"\n  Query [{qid}] {qtext}")

        try:
            start = time.time()
            resp = client.post(
                "/api/query",
                json=query_payload(qtext, collection),
                timeout=QUERY_TIMEOUT,
            )
            elapsed = time.time() - start

            if resp.status_code != 200:
                check(f"[{qid}] query request", False, http_error_detail(resp))
                continue

            data = resp.json()
            retrieved = data.get("retrieved_documents", [])
            reranked = data.get("reranked_documents", [])
            hyde_document = data.get("hyde_document")
            label = data.get("classification_label")

            print(f"    elapsed={elapsed:.1f}s label={label} retrieved={len(retrieved)} reranked={len(reranked)}")
            if hyde_document:
                print(f"    hyde={safe_preview(hyde_document, 120)}")
            print_documents("retrieved_documents", retrieved)
            print_documents("reranked_documents", reranked)

            check(f"[{qid}] hyde returned", bool(hyde_document))
            check(f"[{qid}] retrieved documents", len(retrieved) > 0, str(len(retrieved)))
            check(f"[{qid}] reranked documents", len(reranked) > 0, str(len(reranked)))

            query_success += 1

            if qid in qrel_map:
                query_with_qrels += 1
                relevant_titles = [title_map.get(doc_id, "") for doc_id in qrel_map[qid]]
                retrieved_text = "\n".join(doc.get("content", "") for doc in retrieved)
                hit = any(title and title in retrieved_text for title in relevant_titles)
                check(f"[{qid}] retrieved relevant title", hit)
                if hit:
                    relevant_hit += 1
        except Exception as exc:
            check(f"[{qid}] query request", False, str(exc))

    print("\n  --- Query summary ---")
    print(f"  Successful queries: {query_success}/{len(test_queries_list)}")
    if query_with_qrels:
        print(f"  Queries hitting known relevant title: {relevant_hit}/{query_with_qrels}")


def cleanup_local_temp() -> None:
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
        check("Local temporary files cleaned", True)
    else:
        check("Local temporary files cleaned", True, "nothing to remove")


def test_cleanup(server_url: str, collection: str) -> None:
    print("\n[Step 5] Cleanup")
    try:
        from pymilvus import MilvusClient

        parsed = urlparse(server_url)
        milvus_uri = f"http://{parsed.hostname}:19530"
        client = MilvusClient(uri=milvus_uri, timeout=5)
        if client.has_collection(collection):
            client.drop_collection(collection)
            check(f"Milvus collection '{collection}' cleaned", True)
        else:
            check(f"Milvus collection '{collection}' cleaned", True, "nothing to remove")
    except Exception as exc:
        check(f"Milvus collection '{collection}' cleaned", False, str(exc))

    cleanup_local_temp()


def main() -> None:
    args = parse_args()
    server_url = get_server_url(args.server_url)

    print("=" * 60)
    print("RAG staging smoke test")
    print("Dataset: BeIR/scifact")
    print(f"Server: {server_url}")
    print(f"Collection: {args.collection}")
    print(f"Max docs: {args.max_docs}")
    print(f"Max queries: {args.max_queries}")
    print("=" * 60)

    client = httpx.Client(base_url=server_url, timeout=QUERY_TIMEOUT)

    try:
        if not test_health(client):
            print("\nServer is not ready.")
            sys.exit(1)

        corpus_ds, queries_ds, qrels_ds = download_scifact()
        if corpus_ds is None:
            print("\nFailed to download SciFact.")
            sys.exit(1)

        if not test_index(client, corpus_ds, args.collection, args.max_docs):
            print("\nIndexing failed.")
            sys.exit(1)

        test_queries(client, queries_ds, qrels_ds, corpus_ds, args.max_queries, args.collection)
    finally:
        client.close()
        test_cleanup(server_url, args.collection)

    print("\n" + "=" * 60)
    print(f"Test result: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
