"""
基础 RAG 问答链路（手写版）
在线完整流程：
    用户问题
      → ① 向量检索 Top K 相关文档块
      → ② 拼接成上下文
      → ③ 组装 Prompt（系统约束 + 资料 + 问题）
      → ④ 调用 DeepSeek 生成答案
Java 类比：相当于一个 Service 方法，依次调用检索 DAO → 组装 DTO → 调用远程 LLM 接口
"""
from clients.deepseek_client import deepseek_client
from vectorstores.faiss_store import faiss_store

# RAG 标准提示词模板（面试重点：模板四要素 = 角色 + 上下文 + 约束 + 输出要求）
RAG_SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。请严格根据下面提供的【参考资料】回答用户问题。

要求：
1. 答案必须完全来自参考资料，禁止编造资料中没有的信息
2. 如果参考资料中没有相关内容，请直接回答"根据现有知识库，我无法回答该问题"
3. 回答简洁准确，必要时分点说明

【参考资料】
{context}
"""


class RAGQaChain:
    """RAG 问答链路封装"""

    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def _retrieve(self, question: str):
        """① 检索：返回与问题最相关的文档块"""
        return faiss_store.similarity_search(question, top_k=self.top_k)

    @staticmethod
    def _build_context(docs) -> str:
        """② 上下文组装：把多个文档块拼成一段文本"""
        return "\n\n".join(doc.page_content for doc in docs)

    def answer(self, question: str, stream: bool = False):
        """RAG 问答主入口，stream=True 时返回生成器"""
        # ① 检索
        docs = self._retrieve(question)
        # ② 组装上下文
        context = self._build_context(docs)
        # ③ 构造消息
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": question},
        ]
        # ④ 调用大模型
        # RAG 场景 temperature 调低（0.1~0.3）：让回答稳定、忠于资料，而不是自由发挥
        return deepseek_client.chat(messages, temperature=0.3, stream=stream)

    def answer_with_sources(self, question: str):
        """
        返回答案 + 引用来源（面试加分项：可溯源）
        企业知识库产品（如 FastGPT、Dify）都要求给出引用了哪些文档
        """
        docs = self._retrieve(question)
        context = self._build_context(docs)
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": question},
        ]
        answer = deepseek_client.chat(messages, temperature=0.3)
        # 整理来源：文件名 + 片段
        sources = [
            {
                "source": doc.metadata.get("source", "未知"),
                "snippet": doc.page_content[:100],
            }
            for doc in docs
        ]
        return {"answer": answer, "sources": sources}


# 全局单例
rag_qa_chain = RAGQaChain()
