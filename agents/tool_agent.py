"""
工具调用 Agent（手写版）
循环逻辑：
  ① 带着工具清单请求模型
  ② 模型返回 tool_calls → 本地执行工具 → 结果以 role=tool 追加 → 回到 ①
  ③ 模型不再调用工具 → 返回最终回答
"""
import json
from openai import OpenAI

from config.settings import settings
from tools.schemas import TOOL_SCHEMAS
from tools.registry import tool_registry


class ToolAgent:
    def __init__(self, max_iterations: int = 5):
        self.client = OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
            timeout=60,
        )
        self.model = settings.DEEPSEEK_MODEL
        self.max_iterations = max_iterations  # 防止模型反复调工具陷入死循环

    def run(self, question: str, system_prompt: str = None) -> str:
        system_prompt = system_prompt or "你是一个能使用工具的智能助手，请根据需要调用工具来准确回答问题。"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        for i in range(self.max_iterations):
            # ① 请求模型（带上工具清单）
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",   # auto = 让模型自己决定是否调用
                temperature=0.3,
            )
            msg = resp.choices[0].message
            # ② assistant 消息必须原样存回去（内部带 tool_calls），否则下轮接口报错
            messages.append(msg)

            # ③ 没有工具调用，说明这是最终回答
            if not msg.tool_calls:
                return msg.content

            # ④ 逐个执行工具调用（模型可能一次返回多个，即并行调用）
            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                print(f"[第{i+1}轮] 调用工具：{name}，参数：{args}")
                result = tool_registry.execute(name, **args)
                # 工具结果必须以 role=tool 回传，并带上对应的 tool_call_id
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        raise RuntimeError("工具调用超过最大迭代次数，可能陷入循环")


tool_agent = ToolAgent()
