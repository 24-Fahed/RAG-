"""RAG 客户端 - 用于查询和文档提交的命令行接口。

用法：
    python client.py query "什么是 RAG？"
    python client.py query --interactive
    python client.py index ./documents/
"""

import argparse
import sys
import os

import httpx
from client.config import RAG_SERVER_URL


def query_single(query: str, **kwargs):
    """向 RAG 服务器发送单次查询。"""
    resp = httpx.post(f"{RAG_SERVER_URL}/api/query", json={"query": query, **kwargs}, timeout=120.0)
    resp.raise_for_status()
    data = resp.json()

    print(f"\n{'=' * 60}")
    print(f"Query:  {query}")
    print(f"Answer: {data['answer']}")
    print(f"{'=' * 60}")
    print(f"Retrieved: {len(data.get('retrieved_documents', []))} docs")
    print(f"Reranked:  {len(data.get('reranked_documents', []))} docs")
    if data.get("hyde_document"):
        print(f"HyDE:      {data['hyde_document'][:100]}...")


def interactive_mode(**kwargs):
    """交互式查询循环。"""
    print("=" * 60)
    print("RAG Interactive Client")
    print(f"Server: {RAG_SERVER_URL}")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        try:
            query = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if not query:
            continue

        try:
            query_single(query, **kwargs)
        except httpx.ConnectError:
            print(f"ERROR: Cannot connect to {RAG_SERVER_URL}")
        except httpx.HTTPStatusError as e:
            print(f"ERROR: Server returned {e.response.status_code}: {e.response.text[:200]}")
        except Exception as e:
            print(f"ERROR: {e}")


def index_documents(data_path: str, collection: str = "rag_collection"):
    """将文档上传到 RAG 服务器进行索引。"""
    if not os.path.exists(data_path):
        print(f"ERROR: Path not found: {data_path}")
        return

    files = []
    if os.path.isfile(data_path):
        files.append(open(data_path, "rb"))
    else:
        for fname in os.listdir(data_path):
            fpath = os.path.join(data_path, fname)
            if os.path.isfile(fpath):
                files.append(open(fpath, "rb"))

    if not files:
        print("ERROR: No files found to upload.")
        return

    try:
        resp = httpx.post(
            f"{RAG_SERVER_URL}/api/index",
            files=[("files", f) for f in files],
            data={"collection": collection},
            timeout=300.0,
        )
        resp.raise_for_status()
        data = resp.json()
        print(f"Status:  {data['status']}")
        print(f"Collection: {data['collection']}")
        print(f"Documents:  {data['document_count']}")
        if data.get("message"):
            print(f"Message: {data['message']}")
    finally:
        for f in files:
            f.close()


def main():
    parser = argparse.ArgumentParser(description="RAG Client")
    sub = parser.add_subparsers(dest="command")

    # 查询
    q = sub.add_parser("query", help="查询 RAG 知识库")
    q.add_argument("query", nargs="?", default=None, help="查询字符串")
    q.add_argument("--interactive", "-i", action="store_true", help="交互模式")

    # 索引
    idx = sub.add_parser("index", help="上传文档到知识库")
    idx.add_argument("path", help="文件或目录路径")
    idx.add_argument("--collection", default="rag_collection")

    args = parser.parse_args()

    if args.command == "query":
        if args.interactive or args.query is None:
            interactive_mode()
        else:
            query_single(args.query)
    elif args.command == "index":
        index_documents(args.path, args.collection)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
