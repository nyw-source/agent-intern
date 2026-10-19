"""
第 2 章测试：文档入库 + 向量检索
"""
from config.settings import settings
from vectorstores.faiss_store import faiss_store


def main():
    settings.check()

    # 1. 清空并重建向量库
    faiss_store.clear()
    print("正在解析文档并向量化入库...")
    faiss_store.add_file("docs/test.txt")
    print("文档入库完成！\n")

    # 2. 测试检索（改成你 test.txt 里确实会出现的内容）
    query = "RAG是怎么解决大模型幻觉问题的？"
    print(f"检索问题：{query}\n")
    results = faiss_store.similarity_search(query, top_k=2)

    print("===== 检索结果 Top 2 =====")
    for i, doc in enumerate(results, 1):
        print(f"\n--- 第 {i} 块 ---")
        print(f"来源：{doc.metadata.get('source', '未知')}")
        print(f"内容：{doc.page_content[:200]}...")


if __name__ == "__main__":
    main()
