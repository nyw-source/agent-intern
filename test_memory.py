"""
第 6 章测试：
1. 同一 thread_id 多轮对话能记住上文
2. 不同 thread_id 记忆互相隔离
3. 查看一个会话里存了哪些历史消息
"""
from langchain_core.messages import HumanMessage

from config.settings import settings
from agents.memory_agent import memory_agent


def main():
    settings.check()

    # 会话 A：用固定 thread_id 标识
    config_a = {"configurable": {"thread_id": "user-001"}}

    print("=" * 55)
    r1 = memory_agent.invoke(
        {"messages": [HumanMessage(content="我叫张小明，是软件工程专业的大三学生。")]},
        config_a,
    )
    print("第1轮：", r1["messages"][-1].content)

    r2 = memory_agent.invoke(
        {"messages": [HumanMessage(content="我叫什么名字？我是什么专业？")]},
        config_a,
    )
    print("第2轮：", r2["messages"][-1].content)

    # 会话 B：换一个 thread_id，应该完全不知道 A 的信息
    print("\n" + "=" * 55)
    config_b = {"configurable": {"thread_id": "user-002"}}
    r3 = memory_agent.invoke(
        {"messages": [HumanMessage(content="我叫什么名字？")]},
        config_b,
    )
    print("新会话(user-002)：", r3["messages"][-1].content)

    # 查看会话 A 内部保存的完整消息历史（记忆的本质）
    print("\n" + "=" * 55)
    print("会话 A 保存的历史消息：")
    snapshot = memory_agent.get_state(config_a)
    for i, msg in enumerate(snapshot.values["messages"]):
        print(f"  [{i}] {msg.__class__.__name__}: {msg.content[:40]}")


if __name__ == "__main__":
    main()
