from prometheus_client import Counter, Histogram

# HTTP layer
HTTP_REQUESTS = Counter(
    "app_http_requests_total", "HTTP requests", ["path", "method", "status"]
)
HTTP_LATENCY = Histogram(
    "app_http_request_duration_seconds", "HTTP request duration (s)", ["path", "method"]
)

# RAG pipeline
RETRIEVAL_LATENCY = Histogram(
    "app_retrieval_stage_seconds", "Retrieval stage duration (s)", ["stage"]  # stage: search|rerank
)
GENERATION_LATENCY = Histogram(
    "app_generation_seconds", "LLM generation duration (s)"
)
RETRIEVAL_CANDIDATES = Histogram(
    "app_retrieval_candidates", "Candidates before rerank", buckets=[1,5,10,20,40,80,160,320]
)

# Tokens (approx.)
TOKENS_INPUT = Counter("app_llm_input_tokens_total", "Estimated input tokens")
TOKENS_OUTPUT = Counter("app_llm_output_tokens_total", "Estimated output tokens")
