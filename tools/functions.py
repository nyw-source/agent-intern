"""
工具函数集合：每个工具就是一个普通 Python 函数
函数的名字、docstring、参数 会被用来生成"工具说明书"
"""
import ast
import operator
from datetime import datetime

from rag.qa_chain import rag_qa_chain

# ---------- 工具 1：安全计算器 ----------
# 白名单：只允许这些运算符，从根上杜绝注入
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    """递归计算 AST 节点，只认数字和白名单运算符"""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"不支持的常量: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _ALLOWED_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError("不支持的运算符")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError("不支持的运算符")
        return op(_safe_eval(node.operand))
    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


def calculator(expression: str) -> str:
    """
    计算数学表达式，支持加减乘除、取模、幂运算和括号
    :param expression: 数学表达式，例如 "(3 + 5) * 2"
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}"


# ---------- 工具 2：当前时间 ----------
def get_current_time() -> str:
    """获取当前的日期、星期和时间，当用户问现在几点、今天日期、星期几 时使用。"""
    return datetime.now().strftime("现在是 %Y年%m月%d日 %H:%M:%S %A")


# ---------- 工具 3：知识库查询（复用第 3 章的 RAG）----------
def query_knowledge_base(question: str) -> str:
    """
    查询知识库中关于 RAG、向量数据库、文本分块等主题的资料
    当用户问专业知识问题时使用
    :param question: 要查询的问题
    """
    return rag_qa_chain.answer(question)
