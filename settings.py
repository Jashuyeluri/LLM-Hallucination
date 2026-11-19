import os

VALID_MODES = ("cascade", "full", "llm")
VALID_POLICIES = ("keep", "remove")


def _flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _number(name, default, cast):
    try:
        return cast(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def enable_nli():
    return _flag("ENABLE_NLI", False)


def nli_model():
    return os.environ.get("NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")


def verifier_mode():
    mode = os.environ.get("VERIFIER_MODE", "cascade").strip().lower()
    return mode if mode in VALID_MODES else "cascade"


def cascade_threshold():
    return _number("CASCADE_THRESHOLD", 0.9, float)


def max_iterations():
    return _number("MAX_ITERATIONS", 3, int)


def max_source_chars():
    return _number("MAX_SOURCE_CHARS", 60000, int)


def max_response_chars():
    return _number("MAX_RESPONSE_CHARS", 12000, int)


def max_upload_bytes():
    return _number("MAX_UPLOAD_BYTES", 5000000, int)


def cors_origins():
    raw = os.environ.get("CORS_ORIGINS", "")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def calibrator_path():
    return os.environ.get("CALIBRATOR_PATH", "calibrator.json")


def review_below():
    return _number("REVIEW_BELOW", 0.5, float)


def rate_limit_per_minute():
    return _number("RATE_LIMIT_PER_MINUTE", 30, int)


def consistency_enabled():
    return _flag("ENABLE_CONSISTENCY", True)


def conflict_threshold():
    return _number("CONFLICT_THRESHOLD", 0.8, float)


def rag_top_k():
    return max(1, min(12, _number("RAG_TOP_K", 4, int)))


def rag_context_chars():
    return max(1000, min(20000, _number("RAG_CONTEXT_CHARS", 6000, int)))


def api_auth_token():
    return os.environ.get("API_AUTH_TOKEN", "").strip()


def guard_enabled():
    return _flag("ENABLE_GUARD", True)


def guard_min_overlap():
    return _number("GUARD_MIN_OVERLAP", 0.2, float)
