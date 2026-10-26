"""
FastAPI 应用入口（类比 Spring Boot 的 @SpringBootApplication 主类）
启动：uvicorn api.main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import chat, knowledge

app = FastAPI(
    title="AI Agent 知识库问答服务",
    description="RAG + Function Calling + LangGraph 的后端接口",
    version="1.0.0",
)

# 跨域配置：前后端分离必备，否则浏览器会拦截跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 生产环境应改成具体前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router)
app.include_router(knowledge.router)


@app.get("/api/health", tags=["系统"], summary="健康检查")
def health():
    return {"status": "ok", "service": "agent-intern"}
