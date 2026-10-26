"""
对话相关接口
"""
from fastapi import APIRouter
from langchain_core.messages import HumanMessage

from api.schemas import ChatRequest, ChatResponse
from agents.memory_agent import memory_agent

from fastapi.responses import StreamingResponse

# 路由前缀，类比 Spring 的 @RequestMapping("/api")
router = APIRouter(prefix="/api", tags=["对话"])


@router.post("/chat", response_model=ChatResponse, summary="多轮对话（带记忆）")
def chat(req: ChatRequest):
    """
    普通对话接口：同一 thread_id 的多次请求共享记忆
    注意这里用 def（同步函数），FastAPI 会自动丢到线程池执行，不阻塞事件循环
    """
    config = {"configurable": {"thread_id": req.thread_id}}
    result = memory_agent.invoke(
        {"messages": [HumanMessage(content=req.message)]},
        config,
    )
    reply = result["messages"][-1].content
    return ChatResponse(reply=reply, thread_id=req.thread_id)


@router.post("/chat/stream", summary="流式对话（打字机效果）")
def chat_stream(req: ChatRequest):
    """
    流式接口：token 边生成边返回（SSE）
    LangGraph 的 stream_mode='messages' 会逐 token 产出
    """
    config = {"configurable": {"thread_id": req.thread_id}}

    def event_generator():
        for msg, metadata in memory_agent.stream(
            {"messages": [HumanMessage(content=req.message)]},
            config=config,
            stream_mode="messages",
        ):
            # 只输出模型节点产生的文本 token，过滤工具调用消息
            if (
                metadata.get("langgraph_node") == "llm_call"
                and msg.content
                and not getattr(msg, "tool_calls", None)
            ):
                yield msg.content

    return StreamingResponse(event_generator(), media_type="text/event-stream")
