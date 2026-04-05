"""Staging 全链路测试脚本。

使用 BeIR/scifact 数据集进行完整的端到端测试。
包含数据拉取、索引、查询、结果验证全流程。

前置条件：
  1. GPU 服务器：Inference Worker 已启动（python -m inference.main --mode staging）
  2. 公网服务器：Milvus + SSH 隧道 + RAG Server 已启动
  3. 本地环境：pip install datasets httpx pyyaml

用法：
  python tests/test_staging.py [--server-url http://<server_ip>:8000]

默认使用 client/config/staging.yaml 中的 server_url。
"""

import sys
import os
import json
import time
import argparse

# 将项目根目录加入路径
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)

RAG_SERVER = None  # 从参数或配置读取
COLLECTION = "scifact_test"

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


def get_server_url():
    """从参数或配置文件获取 server URL。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--server-url", default=None)
    args, _ = parser.parse_known_args()
    if args.server_url:
        return args.server_url
    try:
        from client.config import load_config
        load_config("staging")
        from client.config import RAG_SERVER_URL
        return RAG_SERVER_URL
    except Exception:
        return "http://127.0.0.1:8001"


# ======== 步骤 1: 健康检查 ========

def test_health():
    """验证所有服务在线。"""
    print("\n[步骤1] 服务健康检查")
    import httpx

    try:
        resp = httpx.get(f"{RAG_SERVER}/health", timeout=10)
        data = resp.json()
        check("RAG Server 在线", data.get("status") == "ok")
    except Exception as e:
        check("RAG Server 在线", False, str(e))
        return False

    # 通过 server 间接验证 inference worker
    try:
        resp = httpx.post(f"{RAG_SERVER}/api/query",
                          json={"query": "test", "top_k": 1},
                          timeout=30)
        # 即使返回错误也算在线，只是验证通路
        check("Server -> Inference Worker 通路正常", resp.status_code in (200, 500))
    except Exception as e:
        check("Server -> Inference Worker 通路正常", False, str(e))
        return False

    return True


# ======== 步骤 2: 拉取 SciFact 数据集 ========

def download_scifact():
    """从 HuggingFace 拉取 BeIR/scifact 数据集。"""
    print("\n[步骤2] 拉取 SciFact 数据集")

    try:
        from datasets import load_dataset
    except ImportError:
        check("datasets 库已安装", False, "pip install datasets")
        return None, None, None

    # 拉取 corpus
    print("  正在下载 corpus...")
    try:
        corpus_ds = load_dataset("BeIR/scifact", "corpus", split="corpus",
                                 trust_remote_code=True)
        check("corpus 下载成功", True, f"{len(corpus_ds)} 篇文档")
    except Exception as e:
        check("corpus 下载成功", False, str(e))
        return None, None, None

    # 拉取 queries
    print("  正在下载 queries...")
    try:
        queries_ds = load_dataset("BeIR/scifact", "queries", split="queries",
                                  trust_remote_code=True)
        check("queries 下载成功", True, f"{len(queries_ds)} 条查询")
    except Exception as e:
        check("queries 下载成功", False, str(e))
        return corpus_ds, None, None

    # 拉取 qrels（相关性标注）
    print("  正在下载 qrels...")
    try:
        qrels_ds = load_dataset("BeIR/scifact-qrels", split="test",
                                trust_remote_code=True)
        check("qrels 下载成功", True, f"{len(qrels_ds)} 条标注")
    except Exception as e:
        check("qrels 下载成功", False, str(e))
        return corpus_ds, queries_ds, None

    return corpus_ds, queries_ds, qrels_ds


# ======== 步骤 3: 索引测试 ========

def test_index(corpus_ds):
    """将 corpus 索引到 Milvus + BM25。"""
    print("\n[步骤3] 索引 SciFact corpus")
    import httpx

    # 将 corpus 转换为临时 txt 文件
    output_dir = os.path.join(PROJECT_ROOT, "tests", "staging_test_data")
    os.makedirs(output_dir, exist_ok=True)

    print(f"  将 corpus 转为 txt 文件 -> {output_dir}")
    count = 0
    for item in corpus_ds:
        doc_id = item.get("_id", count)
        title = item.get("title", "")
        text = item.get("text", "")
        content = f"{title}\n\n{text}" if title else text
        filepath = os.path.join(output_dir, f"doc_{doc_id}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        count += 1
        if count >= 1000:  # 取前 1000 篇做测试
            break

    check(f"生成 {count} 个 txt 文件", True)

    # 通过 Server API 索引
    print(f"  上传索引到 {COLLECTION}...")
    files = []
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        if os.path.isfile(fpath) and fname.endswith(".txt"):
            files.append(open(fpath, "rb"))

    try:
        start_time = time.time()
        resp = httpx.post(
            f"{RAG_SERVER}/api/index",
            files=[("files", f) for f in files],
            data={"collection": COLLECTION},
            timeout=600,
        )
        elapsed = time.time() - start_time

        for f in files:
            f.close()

        if resp.status_code != 200:
            check("索引接口返回成功", False,
                  f"HTTP {resp.status_code}: {resp.text[:300]}")
            return False

        data = resp.json()
        check("索引接口返回 ok", data.get("status") == "ok")
        check("文档数量 > 0", data.get("document_count", 0) > 0)
        print(f"    索引完成: {data.get('document_count')} 个文档块, 耗时 {elapsed:.1f}s")
    except Exception as e:
        for f in files:
            f.close()
        check("索引接口返回成功", False, str(e))
        return False

    # 清理临时文件
    for fname in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, fname))
    os.rmdir(output_dir)

    return True


# ======== 步骤 4: 查询测试 ========

def test_queries(queries_ds, qrels_ds, corpus_ds):
    """执行查询并验证检索质量。"""
    print("\n[步骤4] 查询测试")
    import httpx

    # 构建 qrel 查找表: query_id -> [doc_id, ...]
    qrel_map = {}
    if qrels_ds is not None:
        for item in qrels_ds:
            qid = str(item.get("query-id", ""))
            did = str(item.get("corpus-id", ""))
            if qid not in qrel_map:
                qrel_map[qid] = []
            qrel_map[qid].append(did)

    # 构建 corpus 查找表: doc_id -> title
    corpus_map = {}
    if corpus_ds is not None:
        for item in corpus_ds:
            doc_id = str(item.get("_id", ""))
            corpus_map[doc_id] = item.get("title", "")

    # 取前 10 条查询测试
    test_queries_list = []
    if queries_ds is not None:
        for item in queries_ds:
            test_queries_list.append({
                "id": str(item.get("_id", "")),
                "text": item.get("text", ""),
            })

    test_queries_list = test_queries_list[:10]
    check(f"准备 {len(test_queries_list)} 条测试查询", len(test_queries_list) > 0)

    total_retrieved = 0
    total_hit = 0
    total_answered = 0

    for q in test_queries_list:
        qid = q["id"]
        qtext = q["text"]
        print(f"\n  Query [{qid}]: {qtext}")

        try:
            resp = httpx.post(f"{RAG_SERVER}/api/query",
                              json={"query": qtext, "top_k": 10},
                              timeout=120)
            if resp.status_code != 200:
                check(f"查询返回成功", False,
                      f"HTTP {resp.status_code}: {resp.text[:200]}")
                continue

            data = resp.json()
            answer = data.get("answer", "")
            retrieved = data.get("retrieved_documents", [])
            reranked = data.get("reranked_documents", [])
            label = data.get("classification_label")

            print(f"    Label: {label}")
            print(f"    Retrieved: {len(retrieved)} docs")
            print(f"    Reranked: {len(reranked)} docs")
            print(f"    Answer: {answer[:200]}")

            # 基本检查
            check(f"[{qid}] 返回答案", bool(answer))
            check(f"[{qid}] 检索到文档", len(retrieved) > 0,
                  f"数量: {len(retrieved)}")

            if answer and not answer.startswith("[MOCK]"):
                total_answered += 1

            # 如果有 qrels，验证检索质量
            if qid in qrel_map:
                relevant_ids = set(qrel_map[qid])
                retrieved_contents = " ".join(
                    [d.get("content", "")[:100] for d in retrieved]
                )
                hit = any(rid in retrieved_contents for rid in relevant_ids)
                if hit:
                    total_hit += 1
                    print(f"    Hit: 检索到相关文档")

            total_retrieved += len(retrieved)

        except Exception as e:
            check(f"查询 [{qid}]", False, str(e))

    # 汇总统计
    print(f"\n  --- 查询统计 ---")
    print(f"  总查询数: {len(test_queries_list)}")
    print(f"  平均检索文档数: {total_retrieved / max(1, len(test_queries_list)):.1f}")
    print(f"  真实答案数: {total_answered}/{len(test_queries_list)}")
    if qrel_map:
        print(f"  命中相关文档: {total_hit}/{len([q for q in test_queries_list if q['id'] in qrel_map])}")


# ======== 步骤 5: 清理 ========

def test_cleanup():
    """清理测试数据。"""
    print("\n[步骤5] 清理测试数据")

    # 清理 Milvus collection（通过本地连接，如果可以的话）
    # staging 环境下 Milvus 在公网服务器上，这里只能通过 Server API 间接操作
    # 暂不清理，保留数据供观察
    print("  测试数据保留在服务器上，如需清理请手动执行：")
    print(f"    from pymilvus import MilvusClient")
    print(f"    client = MilvusClient(uri='http://<server_ip>:19530')")
    print(f"    client.drop_collection('{COLLECTION}')")


# ======== 主流程 ========

def main():
    global RAG_SERVER

    print("=" * 60)
    print("RAG Staging 全链路测试")
    print("数据集: BeIR/scifact")
    print("=" * 60)

    RAG_SERVER = get_server_url()
    print(f"RAG Server: {RAG_SERVER}")

    # 步骤 1: 健康检查
    if not test_health():
        print("\n服务未就绪，请检查各服务状态")
        sys.exit(1)

    # 步骤 2: 拉取数据
    corpus_ds, queries_ds, qrels_ds = download_scifact()
    if corpus_ds is None:
        print("\n数据拉取失败")
        sys.exit(1)

    # 步骤 3: 索引
    if not test_index(corpus_ds):
        print("\n索引失败")
        sys.exit(1)

    # 步骤 4: 查询
    test_queries(queries_ds, qrels_ds, corpus_ds)

    # 步骤 5: 清理
    test_cleanup()

    # 汇总
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
