"""
FAISS 向量知识库封装（LangChain 1.0 兼容版）
核心功能：文档入库、相似度检索、本地持久化
Java 类比：相当于 Lucene 索引 + 检索工具类
"""
import shutil
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config.settings import embeddings, settings
from utils.document_processor import doc_processor


class FAISSStore:
    def __init__(self):
        self.index_path = Path(settings.FAISS_INDEX_PATH)
        self._store: FAISS | None = None
        # 启动时尝试加载本地已有的索引
        if self.index_path.exists() and (self.index_path / "index.faiss").exists():
            self._load_local()

    def _load_local(self):
        """从本地磁盘加载已保存的向量索引"""
        self._store = FAISS.load_local(
            str(self.index_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

    def create_from_documents(self, documents: list[Document]):
        """用一批文档创建全新的向量库（会覆盖旧的）"""
        self._store = FAISS.from_documents(documents, embeddings)
        self.save()

    def add_documents(self, documents: list[Document]):
        """向已有向量库追加文档"""
        if self._store is None:
            self.create_from_documents(documents)
        else:
            self._store.add_documents(documents)
            self.save()

    def add_file(self, file_path: str):
        """便捷方法：直接把一个文件入库"""
        docs = doc_processor.load_and_split(file_path)
        self.add_documents(docs)

    def similarity_search(self, query: str, top_k: int = 3) -> list[Document]:
        """
        相似度检索：返回和问题语义最相关的 top_k 个文档块
        :param query: 用户问题
        :param top_k: 返回最相关的几条
        """
        if self._store is None:
            raise RuntimeError("向量库为空，请先添加文档")
        return self._store.similarity_search(query, k=top_k)

    def save(self):
        """保存向量索引到本地磁盘"""
        if self._store is None:
            return
        if self.index_path.exists():
            shutil.rmtree(self.index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)  # 自动创建目录
        self._store.save_local(str(self.index_path))

    def clear(self):
        """清空向量库"""
        if self.index_path.exists():
            shutil.rmtree(self.index_path)
        self._store = None

    def as_retriever(self, top_k: int = 3):
        """返回 LangChain 标准检索器接口（Retriever）"""
        if self._store is None:
            raise RuntimeError("向量库为空，请先添加文档")
        return self._store.as_retriever(search_kwargs={"k": top_k})



# 全局单例
faiss_store = FAISSStore()
