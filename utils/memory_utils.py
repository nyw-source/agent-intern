"""
消息历史裁剪工具
策略：保留最近的消息，超出长度丢弃最老的（system 提示始终保留）
"""
from langchain_core.messages import trim_messages, SystemMessage


def _char_token_counter(messages) -> int:
    """
    自定义 token 计数器：用字符数近似 token 数
    为什么不用模型直接计数：
      LangChain 的 ChatOpenAI 只内置 OpenAI 官方模型的 tokenizer，
      不识别 deepseek-flash，会抛 NotImplementedError
    经验换算：中文约 1 个字符 ≈ 1~1.5 token，英文约 4 个字符 ≈ 1 token
    生产环境可换成 tiktoken 或 DeepSeek 官方 tokenizer 做精确计数
    """
    total = 0
    for m in messages:
        content = m.content
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            # 兼容多模态消息（content 是片段列表）
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total += len(part["text"])
    return total


def trim_history(messages, max_tokens: int = 2000, system_prompt: str = None):
    """
    裁剪消息历史，防止上下文超限
    :param messages: 完整历史消息
    :param max_tokens: 保留的最大长度（这里按字符数近似，约等于 2000~3000 token）
    :param system_prompt: 系统提示词，始终保留在最前面
    """
    trimmed = trim_messages(
        messages,
        max_tokens=max_tokens,
        strategy="last",              # 保留最近的，丢弃最老的
        token_counter=_char_token_counter,  # 用自定义字符计数
        include_system=True,          # 保留 system 消息
        allow_partial=False,          # 不允许截断半条消息
        start_on="human",             # 裁剪后第一条必须是 human，避免对话角色错乱
    )
    if system_prompt:
        # 保证 system 提示在最前面
        trimmed = [SystemMessage(content=system_prompt)] + [
            m for m in trimmed if m.__class__.__name__ != "SystemMessage"
        ]
    return trimmed
