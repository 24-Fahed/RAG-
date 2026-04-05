"""索引路由 - 处理文档上传和知识库构建。"""

import os
import shutil
import tempfile
from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel

from server.services.pipeline import run_indexing_pipeline

router = APIRouter()


class IndexResponse(BaseModel):
    status: str
    collection: str
    document_count: int
    message: str = ""


@router.post("/index", response_model=IndexResponse)
async def index_documents(
    files: list[UploadFile] = File(...),
    collection: str = Form("rag_collection"),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(20),
):
    """上传文档并构建知识库。"""
    tmpdir = tempfile.mkdtemp()
    saved_paths = []
    try:
        for f in files:
            path = os.path.join(tmpdir, f.filename)
            with open(path, "wb") as out:
                content = await f.read()
                out.write(content)
            saved_paths.append(path)

        result = run_indexing_pipeline(
            data_path=tmpdir,
            collection_name=collection,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        return IndexResponse(
            status="ok",
            collection=collection,
            document_count=result.get("document_count", 0),
            message=result.get("message", ""),
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
