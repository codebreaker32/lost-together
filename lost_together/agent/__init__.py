"""Agentic Vision layer: an LLM hiker agent that spends look-again actions through OpenCV 5 tools.

Modules:
  tools       OpenCV 5 tools on BGR frames (detect, look closer, compare frames, query the map)
  mcp_server  the same tools served over MCP (stdio), frames referenced by id
  llm         OpenAI-compatible client with retries, a request-hash cache and live/replay/stub modes
  loop        one escalation decision: up to 3 tool turns -> strict JSON, rule-based fallback, trace
  ranger      human control: autonomy dial (advise/act) and escape approvals
  bench       Tier C harness: rule vs LLM on identical escalation points, McNemar, latency and cost
"""
