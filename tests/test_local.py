"""本地全链路测试脚本。

验证从"文档索引"到"查询回答"的完整数据流。
在 mock 模式下运行，无需 GPU，无需真实模型。

前置条件：
  1. Milvus 已启动（docker-compose up -d）
  2. Inference Worker 已启动（python -m inference.main --mode local）
  3. RAG Server 已启动（python -m server.main --mode local）

用法：
  python tests/test_local.py
"""

import sys
import os
import time

# 配置路径
RAG_SERVER = "http://127.0.0.1:8001"
INFERENCE_SERVER = "http://127.0.0.1:8000"
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "local_test_data")
COLLECTION = "rag_collection"

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        msg = f"  FAIL  {name}"
        if detail:
            msg += f" -- {detail}"
        print(msg)


def test_health():
    """步骤1: 健康检查。"""
    print("\n[步骤1] 健康检查")
    import httpx

    try:
        resp = httpx.get(f"{INFERENCE_SERVER}/health", timeout=5)
        data = resp.json()
        check("Inference Worker 在线", data.get("status") == "ok")
        check("Inference Worker 为 mock 模式", data.get("mode") == "mock",
              f"实际 mode={data.get('mode')}")
    except Exception as e:
        check("Inference Worker 在线", False, str(e))
        return False

    try:
        resp = httpx.get(f"{RAG_SERVER}/health", timeout=5)
        check("RAG Server 在线", resp.json().get("status") == "ok")
    except Exception as e:
        check("RAG Server 在线", False, str(e))
        return False

    return True


def test_inference_endpoints():
    """步骤2: 验证 Inference Worker 各端点。"""
    print("\n[步骤2] Inference Worker 端点测试")
    import httpx

    # classify
    try:
        resp = httpx.post(f"{INFERENCE_SERVER}/inference/classify",
                          json={"query": "What is RAG?"}, timeout=10)
        data = resp.json()
        check("classify 返回 label", "label" in data, f"响应: {data}")
    except Exception as e:
        check("classify 返回 label", False, str(e))

    # embed
    try:
        resp = httpx.post(f"{INFERENCE_SERVER}/inference/embed",
                          json={"texts": ["test text"]}, timeout=10)
        data = resp.json()
        check("embed 返回 768 维向量",
              len(data.get("embeddings", [[]])[0]) == 768,
              f"实际维度: {len(data.get('embeddings', [[]])[0])}")
    except Exception as e:
        check("embed 返回 768 维向量", False, str(e))

    # hyde
    try:
        resp = httpx.post(f"{INFERENCE_SERVER}/inference/hyde",
                          json={"query": "What is RAG?"}, timeout=10)
        data = resp.json()
        check("hyde 返回 hypothetical_document",
              bool(data.get("hypothetical_document")))
    except Exception as e:
        check("hyde 返回 hypothetical_document", False, str(e))

    # rerank
    try:
        resp = httpx.post(f"{INFERENCE_SERVER}/inference/rerank", json={
            "query": "What is RAG?",
            "documents": [{"content": "doc1", "score": 0.5, "metadata": {}}],
            "model": "monot5",
            "top_k": 5,
        }, timeout=10)
        data = resp.json()
        check("rerank 返回排序后的文档", len(data.get("documents", [])) > 0)
    except Exception as e:
        check("rerank 返回排序后的文档", False, str(e))

    # compress
    try:
        resp = httpx.post(f"{INFERENCE_SERVER}/inference/compress", json={
            "query": "What is RAG?",
            "context": "RAG is a technique. It combines retrieval and generation.",
            "method": "recomp_extractive",
            "ratio": 0.6,
        }, timeout=10)
        data = resp.json()
        check("compress 返回压缩文本", bool(data.get("compressed")))
    except Exception as e:
        check("compress 返回压缩文本", False, str(e))

    # generate
    try:
        resp = httpx.post(f"{INFERENCE_SERVER}/inference/generate", json={
            "query": "What is RAG?",
            "context": "some context",
            "max_out_len": 50,
        }, timeout=10)
        data = resp.json()
        check("generate 返回 mock 答案",
              data.get("answer", "").startswith("[MOCK]"),
              f"实际: {data.get('answer', '')[:50]}")
    except Exception as e:
        check("generate 返回 mock 答案", False, str(e))


def test_index():
    """步骤3: 索引测试 - 上传文档并验证存储。"""
    print("\n[步骤3] 索引测试")
    import httpx

    if not os.path.exists(TEST_DATA_DIR):
        check("测试数据目录存在", False, f"路径: {TEST_DATA_DIR}")
        return False

    files = []
    for fname in os.listdir(TEST_DATA_DIR):
        fpath = os.path.join(TEST_DATA_DIR, fname)
        if os.path.isfile(fpath):
            files.append(open(fpath, "rb"))

    if not files:
        check("找到测试文件", False, f"目录为空: {TEST_DATA_DIR}")
        return False

    check("找到测试文件", True, f"{len(files)} 个文件")

    try:
        resp = httpx.post(
            f"{RAG_SERVER}/api/index",
            files=[("files", f) for f in files],
            data={"collection": COLLECTION},
            timeout=300,
        )
        for f in files:
            f.close()

        if resp.status_code != 200:
            check("索引接口返回成功", False,
                  f"HTTP {resp.status_code}: {resp.text[:200]}")
            return False

        data = resp.json()
        check("索引接口返回 ok", data.get("status") == "ok",
              f"实际: {data.get('status')}")
        check("文档数量 > 0", data.get("document_count", 0) > 0,
              f"实际: {data.get('document_count')}")
        print(f"    索引结果: {data.get('document_count')} 个文档块已写入 {COLLECTION}")
    except Exception as e:
        for f in files:
            f.close()
        check("索引接口返回成功", False, str(e))
        return False

    return True


def test_query():
    """步骤4: 查询测试 - 验证全链路数据流。"""
    print("\n[步骤4] 查询测试")
    import httpx

    queries = [
        "What is RAG?",
        "How does Milvus work?",
        "What models does the system use?",
    ]

    for q in queries:
        print(f"\n  Query: {q}")
        try:
            resp = httpx.post(f"{RAG_SERVER}/api/query",
                              json={"query": q, "top_k": 5},
                              timeout=120)
            if resp.status_code != 200:
                check(f"查询返回成功", False,
                      f"HTTP {resp.status_code}: {resp.text[:200]}")
                continue

            data = resp.json()
            answer = data.get("answer", "")
            retrieved = data.get("retrieved_documents", [])
            reranked = data.get("reranked_documents", [])
            hyde = data.get("hyde_document")
            label = data.get("classification_label")

            print(f"  Classification label: {label}")
            if hyde:
                print(f"  HyDE: {hyde[:120]}...")
            print(f"  Retrieved: {len(retrieved)} docs")
            print(f"  Reranked: {len(reranked)} docs")
            print(f"  Answer: {answer}")
            if retrieved:
                print(f"  Top doc score: {retrieved[0].get('score', 'N/A')}")
                print(f"  Top doc content: {retrieved[0].get('content', '')[:100]}...")

            check(f"返回答案", bool(answer))
            check(f"检索到文档", len(retrieved) > 0, f"数量: {len(retrieved)}")
            check(f"答案为 mock", answer.startswith("[MOCK]"))
        except Exception as e:
            check(f"查询", False, str(e))


def test_cleanup():
    """步骤5: 清理测试数据。"""
    print("\n[步骤5] 清理测试数据")
    try:
        from pymilvus import MilvusClient
        client = MilvusClient(uri="http://localhost:19530")
        if client.has_collection(COLLECTION):
            client.drop_collection(COLLECTION)
            check("测试 collection 已清理", True)
        else:
            check("测试 collection 已清理", True, "不存在，无需清理")
    except Exception as e:
        check("测试 collection 已清理", False, str(e))

    bm25_dir = os.path.join(os.path.dirname(__file__), "..", "bm25_data")
    bm25_file = os.path.join(bm25_dir, f"{COLLECTION}.pkl")
    if os.path.exists(bm25_file):
        os.remove(bm25_file)
        check("BM25 索引文件已清理", True)
    else:
        check("BM25 索引文件已清理", True, "不存在")


def main():
    print("=" * 60)
    print("RAG 本地全链路测试")
    print("=" * 60)

    if not test_health():
        print("\n服务未就绪，请先启动 Inference Worker 和 RAG Server")
        sys.exit(1)

    try:
        test_inference_endpoints()
        test_index()
        test_query()
    finally:
        test_cleanup()

    print("\n" + "=" * 60)
    print(f"测试结果: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
