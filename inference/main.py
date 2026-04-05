"""推理服务 Worker - FastAPI 应用。

运行在 GPU 服务器上（或在本地 mock 模式下运行）。
模式通过 --mode 参数指定：
  - local:      返回模拟数据，不加载模型，无需 GPU。
  - staging:    部署到真实服务器，用于集成测试和参数调优。
  - production: 与 staging 相同，用于正式生产环境。

用法: python -m inference.main --mode local
"""

import argparse

# 在导入任何依赖配置的模块之前，先解析 --mode 参数
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--mode", default="local", choices=["local", "staging", "production"],
                     help="部署模式: local / staging / production (默认: local)")
_args, _ = _parser.parse_known_args()

from inference.config import load_config
load_config(_args.mode)

# 现在可以安全导入依赖配置的模块
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
    print(f"配置文件: config/{_args.mode}.yaml")
    uvicorn.run(app, host=HOST, port=PORT)
