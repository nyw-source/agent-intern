"""
LangGraph 工作流 Agent（Graph API 手写版）
核心三要素：
  State（状态）：节点之间流转的共享数据
  Node（节点）：每个节点是一个函数，读 state → 返回 state 更新
  Edge（边）：节点的连接，普通边固定走向，条件边按规则分流
"""
from typing import TypedDict, Annotated

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from config.settings import settings
from tools.langchain_tools import lc_tools

# ① 绑定了工具的 LLM
llm = ChatOpenAI(
    model=settings.DEEPSEEK_MODEL,
    api_key=settings.DEEPSEEK_API_KEY,
    base_url=settings.DEEPSEEK_BASE_URL,
    temperature=0.3,
    timeout=60,
)
llm_with_tools = llm.bind_tools(lc_tools)

# 工具名 → 工具对象 的映射
tools_by_name = {t.name: t for t in lc_tools}

SYSTEM_PROMPT = "你是一个能使用工具的智能助手，请根据需要调用工具来准确回答问题。"


# ② 定义图的状态：在所有节点间流转的共享数据
class AgentState(TypedDict):
    # add_messages 是 reducer（归约器）：
    # 节点返回新消息时，它自动把新消息"追加"到列表，而不是覆盖
    messages: Annotated[list[AnyMessage], add_messages]


# ③ 节点 1：调用 LLM（让它决策是否调工具）
def llm_call(state: AgentState):
    response = llm_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )
    return {"messages": [response]}


# ④ 节点 2：执行工具（可能一次并行多个）
def tool_node(state: AgentState):
    last_message = state["messages"][-1]
    result = []
    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        # 每个工具结果包装成 ToolMessage，必须带 tool_call_id
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


# ⑤ 条件边：模型决定下一步去哪
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"   # 模型要调工具 → 去工具节点
    return END           # 模型给出最终回答 → 结束


# ⑥ 构建图
builder = StateGraph(AgentState)
builder.add_node("llm_call", llm_call)
builder.add_node("tools", tool_node)

builder.add_edge(START, "llm_call")                 # 入口先到 llm_call
# 条件边：从 llm_call 出发，由 should_continue 的返回值决定走向
builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {"tools": "tools", END: END},
)
builder.add_edge("tools", "llm_call")              # 工具执行完回到 llm_call，形成循环

# 编译成可执行的图
graph_agent = builder.compile()
