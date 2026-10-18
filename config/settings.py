"""
配置加载模块
类比 Java：相当于 Spring Boot 的 application.yml + @ConfigurationProperties 配置类
启动时从 .env 读取一次，全局共享
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
# 显式指定项目根目录下的 .env，保证无论从哪个目录启动都能读到
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)


class Settings:
    """全局配置类（类比 Java 的配置常量类）"""

    # ---- API 配置 ----
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

    # ---- 生成参数（第 3 章调 RAG 时会用到，先理解含义）----
    TEMPERATURE: float = 0.7   # 越小越稳定确定，越大越发散有创意
    MAX_TOKENS: int = 2048     # 单次回答的最大长度

    @classmethod
    def check(cls):
        """启动前自检：密钥没配好就快速失败，而不是调 API 时才报 401"""
        if not cls.DEEPSEEK_API_KEY or "在这里" in cls.DEEPSEEK_API_KEY:
            raise RuntimeError(
                "未检测到有效 DEEPSEEK_API_KEY，请把 .env.example 复制为 .env 并填入真实密钥"
            )

# 全局复用的 Embedding 实例（符合 LangChain 标准接口，所有组件通用）
embeddings = OpenAIEmbeddings(
    api_key=settings.ZHIPU_API_KEY,
    base_url=settings.ZHIPU_BASE_URL,
    model=settings.EMBEDDING_MODEL,
    dimensions=settings.EMBEDDING_DIM,
)


# 模块级单例，其他文件直接 from config.settings import settings
settings = Settings()
