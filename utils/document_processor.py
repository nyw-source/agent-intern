"""
文档处理工具：加载文件 + 智能分块
核心知识点：
1. 不同格式文档统一加载成 LangChain Document 对象
2. 递归字符分块：按标点、换行、空格逐级切分，保证语义完整
3. 重叠分块：避免关键信息刚好在切块边界被切断
"""
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document


class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        :param chunk_size: 每个文本块的最大字符数
        :param chunk_overlap: 相邻块的重叠字符数（通常为 chunk_size 的 10%）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
        )

    def load_file(self, file_path: str) -> list[Document]:
        """加载单个文档，自动根据后缀选择加载器"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(file_path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(file_path)
        elif suffix == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")

        return loader.load()

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """把长文档切成小块"""
        return self.splitter.split_documents(documents)

    def load_and_split(self, file_path: str) -> list[Document]:
        """一站式：加载 + 分块"""
        docs = self.load_file(file_path)
        return self.split_documents(docs)


# 全局单例（faiss_store.py 里 from utils.document_processor import doc_processor 导入的就是它）
doc_processor = DocumentProcessor()
