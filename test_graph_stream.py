"""
观察 LangGraph 每个节点的执行顺序
stream_mode='updates'：每执行完一个节点，就打印该节点的更新
"""
from langchain_core.messages import HumanMessage
from config.settings import settings
from agents.graph_agent import graph_agent


def main():
    settings.check()
    question = "帮我算一下 (1 + 2) * 3，再告诉我现在几点"

    print(f"问：{question}\n")
    for step in graph_agent.stream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode="updates",
    ):
        for node_name, state_update in step.items():
            print(f"▶ 执行节点：{node_name}")
            for msg in state_update.get("messages", []):
                kind = msg.__class__.__name__
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tools = [tc["name"] for tc in msg.tool_calls]
                    print(f"    [{kind}] 模型决定调用工具：{tools}")
                elif kind == "ToolMessage":
                    print(f"    [{kind}] 工具返回：{msg.content}")
                else:
                    print(f"    [{kind}] 最终回答已生成")


if __name__ == "__main__":
    main()
