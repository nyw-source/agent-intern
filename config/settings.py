"""
配置加载模块
类比 Java：Spring Boot 的 application.yml + @ConfigurationProperties 配置类
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

    # ---- DeepSeek API 配置 ----
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

    # ---- 生成参数 ----
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2048

    # ---- 智谱 Embedding 配置 ----
    ZHIPU_API_KEY: str = os.getenv("ZHIPU_API_KEY", "")
    ZHIPU_BASE_URL: str = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "embedding-3")
    EMBEDDING_DIM: int = 1024

    # ---- 向量库配置 ----
    FAISS_INDEX_PATH: str = str(
        Path(__file__).resolve().parent.parent / "data" / "faiss_index"
    )

    @classmethod
    def check(cls):
        """启动前自检：密钥没配好就快速失败"""
        if not cls.DEEPSEEK_API_KEY or "在这里" in cls.DEEPSEEK_API_KEY:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY")
        if not cls.ZHIPU_API_KEY or "你的" in cls.ZHIPU_API_KEY:
            raise RuntimeError("未配置 ZHIPU_API_KEY，请在 .env 中填入智谱密钥")


# ① 先实例化配置单例
settings = Settings()

# ② 再创建 Embedding 实例（必须在 settings 之后，因为要读取它的属性）
# 全局复用，符合 LangChain 标准 Embeddings 接口，向量库/检索器/Agent 都能直接用
embeddings = OpenAIEmbeddings(
    api_key=settings.ZHIPU_API_KEY,
    base_url=settings.ZHIPU_BASE_URL,
    model=settings.EMBEDDING_MODEL,
    dimensions=settings.EMBEDDING_DIM,
)
