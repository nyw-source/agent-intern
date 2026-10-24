"""
工具的 JSON Schema（OpenAI tools 格式）
这是发给模型的"工具说明书"，模型靠它判断：要不要调、调哪个、参数填什么
"""
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式，支持加减乘除、取模、幂运算和括号。当用户需要算术计算时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如 '(3 + 5) * 2'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期、星期和时间。当用户问现在几点、今天星期几、当前日期时使用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_base",
            "description": "查询知识库中关于 RAG、向量数据库、文本分块等主题的资料。当用户问专业知识问题时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "要查询的问题",
                    }
                },
                "required": ["question"],
            },
        },
    },
]
