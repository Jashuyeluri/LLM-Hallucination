import contextlib
import contextvars
import hashlib
import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

_tracker = contextvars.ContextVar("llm_call_tracker", default=None)


@contextlib.contextmanager
def track_calls():
    counter = {"calls": 0}
    token = _tracker.set(counter)
    try:
        yield counter
    finally:
        _tracker.reset(token)


def _default_model():
    if LLM_PROVIDER == "groq":
        return GROQ_MODEL
    if LLM_PROVIDER == "ollama":
        return OLLAMA_MODEL
    raise ValueError("LLM_PROVIDER must be either 'ollama' or 'groq'.")


def _cache_path(prompt, model):
    directory = os.environ.get("LLM_CACHE_DIR", "").strip()
    if not directory:
        return None
    key = hashlib.sha256(json.dumps([LLM_PROVIDER, model, prompt]).encode("utf-8")).hexdigest()
    return Path(directory) / f"{key}.txt"


def chat(prompt, model=None):
    counter = _tracker.get()
    if counter is not None:
        counter["calls"] += 1
    resolved = model or _default_model()
    path = _cache_path(prompt, resolved)
    if path is not None and path.exists():
        return path.read_text(encoding="utf-8")
    text = _send(prompt, resolved)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return text


def _send(prompt, model):
    if LLM_PROVIDER == "groq":
        return _chat_groq(prompt, model)
    if LLM_PROVIDER == "ollama":
        return _chat_ollama(prompt, model)
    raise ValueError("LLM_PROVIDER must be either 'ollama' or 'groq'.")


def _chat_ollama(prompt, model):
    import ollama
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return response["message"]["content"].strip()


def _chat_groq(prompt, model):
    import time
    from groq import Groq, RateLimitError
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys")
    client = Groq(api_key=GROQ_API_KEY)
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0,
            )
            return (response.choices[0].message.content or "").strip()
        except RateLimitError:
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))
