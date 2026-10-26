"""
知识库接口：RAG 问答 + 文档上传入库
"""
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas import (
    DocumentUploadResponse,
    RAGChatRequest,
    RAGChatResponse,
    Source,
)
from rag.qa_chain import rag_qa_chain
from utils.document_processor import doc_processor
from vectorstores.faiss_store import faiss_store

router = APIRouter(prefix="/api/rag", tags=["知识库"])

ALLOWED_SUFFIXES = {".txt", ".pdf", ".docx"}


@router.post("/chat", response_model=RAGChatResponse, summary="知识库问答")
def rag_chat(req: RAGChatRequest):
    """基于知识库回答问题，并返回引用来源"""
    result = rag_qa_chain.answer_with_sources(req.question)
    return RAGChatResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
    )


@router.post("/documents/upload", response_model=DocumentUploadResponse, summary="上传文档入库")
async def upload_document(file: UploadFile = File(...)):
    """
    上传 txt/pdf/docx 文档：临时保存 → 解析分块 → 入向量库 → 删除临时文件
    涉及文件读取用 async def
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        # HTTPException 会自动返回标准错误格式
        raise HTTPException(status_code=400, detail="仅支持 txt/pdf/docx 格式")

    content = await file.read()

    # 写到临时文件（文档加载器按文件路径读取）
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        docs = doc_processor.load_and_split(tmp_path)
        faiss_store.add_documents(docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档入库失败：{e}")
    finally:
        # 无论成功失败都清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return DocumentUploadResponse(
        message="文档入库成功",
        filename=file.filename,
        chunks=len(docs),
    )
