"""
RAG 问答链路（LangChain LCEL 标准写法）
LCEL：LangChain Expression Language，用 | 把组件像管道一样串联
Java 类比：类似 Stream API 的链式调用 / Unix 管道，每个组件实现统一 Runnable 接口
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config.settings import settings
from rag.qa_chain import RAG_SYSTEM_PROMPT
from vectorstores.faiss_store import faiss_store

# ① 大模型（注意：这里用的是 LangChain 的 ChatOpenAI，不是我们手写的 DeepSeekClient）
llm = ChatOpenAI(
    model=settings.DEEPSEEK_MODEL,
    api_key=settings.DEEPSEEK_API_KEY,
    base_url=settings.DEEPSEEK_BASE_URL,
    temperature=0.3,
    timeout=60,
)

# ② 标准检索器
retriever = faiss_store.as_retriever(top_k=3)

# ③ Prompt 模板（{context} 和 {question} 是占位符，管道运行时自动填充）
prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("user", "{question}"),
])


# ④ 文档格式化函数（Runnable 也可以是普通函数）
def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# ⑤ LCEL 管道：输入问题 → 并行准备 context/question → 套模板 → 调 LLM → 解析成字符串
chain = (
    {
        "context": retriever | format_docs,   # 检索结果再格式化
        "question": RunnablePassthrough(),    # 原始问题原样透传
    }
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    print(chain.invoke("RAG是怎么解决幻觉问题的？"))
