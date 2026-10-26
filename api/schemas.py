"""
请求/响应数据模型（DTO）
继承 BaseModel 后，FastAPI 自动校验字段、自动生成 OpenAPI 文档
"""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户消息")
    thread_id: str = Field(default="default", description="会话ID，同一ID共享记忆")


class ChatResponse(BaseModel):
    reply: str = Field(description="Agent回复")
    thread_id: str = Field(description="会话ID")


class RAGChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="知识库问题")


class Source(BaseModel):
    source: str = Field(description="来源文档")
    snippet: str = Field(description="匹配片段")


class RAGChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class DocumentUploadResponse(BaseModel):
    message: str
    filename: str
    chunks: int = Field(description="切分后的文档块数量")
