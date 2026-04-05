"""RAG 服务端 - FastAPI 应用。

运行在公网服务器上。接收客户端请求，编排 RAG 流水线，
将 ML 推理任务委托给 GPU Worker，并在本地管理 Milvus + BM25。
"""

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
    uvicorn.run("server.main:app", host=HOST, port=PORT, reload=True)
