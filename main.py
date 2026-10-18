"""
第 1 章运行入口
用法：
    python main.py            # 非流式（一次性返回完整回答）
    python main.py --stream   # 流式（打字机效果）
"""
import sys

from config.settings import settings
from clients.deepseek_client import deepseek_client


def main():
    # 1. 启动前检查密钥
    settings.check()

    # 2. 构造 messages（system 定人设，user 是提问）
    messages = [
        {"role": "system", "content": "你是一名资深 AI Agent 后端工程师，回答简洁、注重原理。"},
        {"role": "user", "content": "用三句话解释什么是大模型 Function Calling。"},
    ]

    # 3. 按命令行参数选择流式 / 非流式
    if "--stream" in sys.argv:
        print("【流式输出】\n")
        for piece in deepseek_client.chat(messages, stream=True):
            print(piece, end="", flush=True)  # flush=True 才能看到逐字打印
        print("\n")
    else:
        print("【非流式输出】\n")
        answer = deepseek_client.chat(messages)
        print(answer)


if __name__ == "__main__":
    main()
