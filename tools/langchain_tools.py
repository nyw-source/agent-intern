"""
LangChain 标准工具写法
@tool 装饰器：自动把函数包装成符合 LangChain 标准接口的工具
完整的 Agent 循环编排我们在第 5 章用 LangGraph 实现
"""
from langchain_core.tools import tool

from tools import functions


@tool
def calculator(expression: str) -> str:
    """计算数学表达式，支持加减乘除、取模、幂运算和括号。"""
    return functions.calculator(expression)


@tool
def get_current_time() -> str:
    """获取当前日期、星期和时间。"""
    return functions.get_current_time()


@tool
def query_knowledge_base(question: str) -> str:
    """查询知识库中关于 RAG、向量数据库、文本分块等主题的资料。"""
    return functions.query_knowledge_base(question)


# 标准工具列表（可直接 bind_tools 或交给 LangGraph）
lc_tools = [calculator, get_current_time, query_knowledge_base]


if __name__ == "__main__":
    # 查看自动生成的 schema，和我们手写的 schemas.py 本质一样
    print(calculator.tool_call_schema.model_json_schema())
    print("工具数量:", len(lc_tools))
