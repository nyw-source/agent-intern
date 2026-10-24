"""
第 4 章测试：分别触发三种工具 + 一次闲聊
"""
from config.settings import settings
from agents.tool_agent import tool_agent


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
        print(f"答：{tool_agent.run(q)}")


if __name__ == "__main__":
    main()
