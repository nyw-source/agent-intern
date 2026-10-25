"""
带记忆的 Agent：编译时注入内存检查点
MemorySaver：把状态快照存在内存里，进程重启后记忆消失（适合演示/开发）
"""
from langgraph.checkpoint.memory import MemorySaver

from agents.graph_agent import build_agent_graph

# 内存检查点（单例，整个应用共用）
memory_checkpointer = MemorySaver()

# 编译出带记忆的图
memory_agent = build_agent_graph(checkpointer=memory_checkpointer)
