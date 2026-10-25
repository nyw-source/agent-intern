"""
持久化记忆 Agent：用 SQLite 把检查点写到本地磁盘
重启程序后，同一个 thread_id 的历史依然在
"""
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from agents.graph_agent import build_agent_graph

# 数据库文件存在 data 目录下（check_same_thread=False 允许跨线程使用）
conn = sqlite3.connect("data/agent_memory.db", check_same_thread=False)
sqlite_checkpointer = SqliteSaver(conn)

persistent_agent = build_agent_graph(checkpointer=sqlite_checkpointer)
