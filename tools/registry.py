"""
工具注册中心
类比 Java：一个 Map<String, Function>，按名字注册 Bean，运行时按名字查找执行
"""
from tools import functions

# 工具名 → 函数 的映射表
_REGISTRY = {
    "calculator": functions.calculator,
    "get_current_time": functions.get_current_time,
    "query_knowledge_base": functions.query_knowledge_base,
}


class ToolRegistry:
    def execute(self, name: str, **kwargs) -> str:
        """按名字执行工具；工具自身报错不能让整个 Agent 崩溃，要兜底返回"""
        func = _REGISTRY.get(name)
        if func is None:
            return f"错误：未找到工具 {name}"
        try:
            return str(func(**kwargs))
        except Exception as e:
            return f"工具 {name} 执行出错：{e}"


tool_registry = ToolRegistry()
