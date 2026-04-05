"""RAG 服务端 - FastAPI 应用。

运行在公网服务器上。接收客户端请求，编排 RAG 流水线，
将 ML 推理任务委托给 GPU Worker，并在本地管理 Milvus + BM25。

用法: python -m server.main --mode staging
"""

import argparse

# 在导入任何依赖配置的模块之前，先解析 --mode 参数
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--mode", default="local", choices=["local", "staging", "production"],
                     help="部署模式: local / staging / production (默认: local)")
_args, _ = _parser.parse_known_args()

from server.config import load_config
load_config(_args.mode)

# 现在可以安全导入依赖配置的模块
from fastapi import FastAPI
from server.config import HOST, PORT
from server.routers import query, indexing

app = FastAPI(title="RAG Server", version="0.1.0")

app.include_router(query.router, prefix="/api", tags=["query"])
app.include_router(indexing.router, prefix="/api", tags=["indexing"])


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    reload = _args.mode != "production"
    print(f"RAG 服务端正在启动，地址 {HOST}:{PORT}")
    print(f"配置文件: config/{_args.mode}.yaml")
    uvicorn.run("server.main:app", host=HOST, port=PORT, reload=reload)
