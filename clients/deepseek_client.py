"""
DeepSeek 客户端封装
核心知识点：
1. DeepSeek 完全兼容 OpenAI 协议：用 openai SDK，只改 base_url 和 api_key
2. 模块级单例（类比 Java Spring 容器里的 @Bean）
3. 同时支持非流式 / 流式（SSE）两种调用
4. tenacity 做指数退避重试，且只重试"值得重试"的错误
"""
from openai import (
    OpenAI,
    APIConnectionError,   # 网络连不上
    APITimeoutError,      # 请求超时
    RateLimitError,       # 429 限流
    InternalServerError,  # 5xx 服务端错误
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config.settings import settings

# 只有这四类错误值得重试：限流、超时、连接失败、服务端故障
# 401 密钥错误、400 参数错误属于"重试一万次也不会成功"，直接抛出
RETRYABLE_ERRORS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)


class DeepSeekClient:
    """DeepSeek 对话客户端（类比 Java 里封装好的 Service 类）"""

    def __init__(self):
        # 关键：DeepSeek 走 OpenAI 兼容协议，唯一区别就是 base_url
        self._client = OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
            timeout=60.0,
        )
        self.model = settings.DEEPSEEK_MODEL

    @retry(
        stop=stop_after_attempt(3),                       # 最多重试 3 次
        wait=wait_exponential(multiplier=1, min=2, max=10),  # 等待 2s → 4s → 8s
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        reraise=True,                                     # 重试耗尽后抛出原始异常
    )
    def chat(self, messages: list[dict], temperature: float = None, stream: bool = False):
        """统一对话入口，stream=True 时返回生成器（逐块产出）"""
        if stream:
            return self._chat_stream(messages, temperature)
        return self._chat_once(messages, temperature)

    def _chat_once(self, messages: list[dict], temperature: float) -> str:
        """非流式：模型把完整回答生成完，一次性返回"""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=settings.TEMPERATURE if temperature is None else temperature,
            max_tokens=settings.MAX_TOKENS,
        )
        return resp.choices[0].message.content

    def _chat_stream(self, messages: list[dict], temperature: float):
        """流式：返回生成器，每生成几个 token 就 yield 一块"""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=settings.TEMPERATURE if temperature is None else temperature,
            max_tokens=settings.MAX_TOKENS,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:  # 最后一个 chunk 的 content 可能是 None，需判空
                yield delta


# 模块级单例：整个项目共用一个 client（底层复用 HTTP 连接池）
deepseek_client = DeepSeekClient()
