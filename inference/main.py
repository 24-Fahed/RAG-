"""推理服务 Worker - FastAPI 应用。

运行在 GPU 服务器上（或在本地 mock 模式下运行）。
模式由 deploy.yaml 控制：
  - mock:       返回模拟数据，不加载模型，无需 GPU。
  - staging:    部署到真实服务器，用于集成测试和参数调优。
  - production: 与 staging 相同，用于正式生产环境。
"""

from fastapi import FastAPI
from inference.config import HOST, PORT, MODE

app = FastAPI(title="RAG Inference Worker", version="0.1.0")

if MODE == "mock":
    from inference.mock_routers import router
    app.include_router(router, prefix="/inference")
else:
    from inference.routers import classify, embed, rerank, compress, generate
    app.include_router(classify.router, prefix="/inference", tags=["classify"])
    app.include_router(embed.router, prefix="/inference", tags=["embed"])
    app.include_router(rerank.router, prefix="/inference", tags=["rerank"])
    app.include_router(compress.router, prefix="/inference", tags=["compress"])
    app.include_router(generate.router, prefix="/inference", tags=["generate"])


@app.get("/health")
def health():
    return {"status": "ok", "mode": MODE}


if __name__ == "__main__":
    import uvicorn
    print(f"推理服务 Worker 正在以 [{MODE}] 模式启动，地址 {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
