"""
第 3 章测试：RAG 知识库问答
"""
from config.settings import settings
from rag.qa_chain import rag_qa_chain


def main():
    settings.check()

    # 前两个问题知识库中有答案，第三个问题知识库中完全没有
    questions = [
        "RAG是怎么解决大模型幻觉问题的？",
        "常见的文本分块策略有哪些？",
        "公司明年的营收目标是多少？",
    ]

    for q in questions:
        print(f"\n{'='*55}")
        print(f"问：{q}")
        answer = rag_qa_chain.answer(q)
        print(f"答：{answer}")

    # 额外验证：引用来源
    print(f"\n{'='*55}")
    print("【引用来源演示】")
    result = rag_qa_chain.answer_with_sources("FAISS和Milvus有什么区别？")
    print(f"答：{result['answer']}")
    print("来源：")
    for s in result["sources"]:
        print(f"  - {s['source']}：{s['snippet'][:40]}...")


if __name__ == "__main__":
    main()
