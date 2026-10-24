"""
第 5 章测试：LangGraph 工作流 Agent
"""
from langchain_core.messages import HumanMessage

from config.settings import settings
from agents.graph_agent import graph_agent


def main():
    settings.check()
    questions = [
        "帮我算一下 (123 + 456) * 2 等于多少？",
        "现在几点了？今天星期几？",
        "RAG是怎么解决幻觉问题的？",
        "你好，你是谁？",
    ]

    for q in questions:
        print(f"\n{'='*55}")
        print(f"问：{q}")
        # 输入和输出都以 {"messages": [...]} 为标准结构
        result = graph_agent.invoke({"messages": [HumanMessage(content=q)]})
        print(f"答：{result['messages'][-1].content}")


if __name__ == "__main__":
    main()
