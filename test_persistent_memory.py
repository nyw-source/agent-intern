"""
验证持久化记忆：第一次运行记住信息，可注释掉第一轮、重启后再问
"""
from langchain_core.messages import HumanMessage
from config.settings import settings
from agents.persistent_memory_agent import persistent_agent

config = {"configurable": {"thread_id": "persist-user-001"}}
settings.check()

# 第一次运行后，可把下面这行注释掉，重新运行脚本，验证记忆仍在
r1 = persistent_agent.invoke(
    {"messages": [HumanMessage(content="记住：我的项目编号是 A-2026-007。")]},
    config,
)
print("写入：", r1["messages"][-1].content)

r2 = persistent_agent.invoke(
    {"messages": [HumanMessage(content="我的项目编号是多少？")]},
    config,
)
print("读取：", r2["messages"][-1].content)
