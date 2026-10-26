import httpx

BASE = "http://127.0.0.1:8000"

# 1. 健康检查
print("health:", httpx.get(f"{BASE}/api/health").json())

# 2. 普通对话（验证记忆）
body = {"message": "我叫李雷，记住了吗", "thread_id": "test-001"}
httpx.post(f"{BASE}/api/chat", json=body)
r = httpx.post(f"{BASE}/api/chat", json={"message": "我叫什么？", "thread_id": "test-001"})
print("chat:", r.json())

# 3. RAG 问答
r = httpx.post(f"{BASE}/api/rag/chat", json={"question": "RAG怎么解决幻觉问题？"})
print("rag:", r.json()["answer"][:80])

# 4. 流式对话
with httpx.stream("POST", f"{BASE}/api/chat/stream",
                  json={"message": "用三句话介绍你自己", "thread_id": "stream-001"},
                  timeout=60) as resp:
    print("stream:", resp.read().decode()[:200])
