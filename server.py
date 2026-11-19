import time
import logging
import os
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import calibration
import settings
from file_utils import read_uploaded_bytes
from json_utils import InvalidLLMOutput
from nli_verifier import is_available
from pipeline import run_pipeline
from source_ingestion import combine_sources, fetch_url_text

app = FastAPI(title="Hallucination Verification Lab")

_origins = settings.cors_origins()
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

_hits = defaultdict(deque)
_redis_client = None
_redis_checked = False
logger = logging.getLogger("uvicorn.error")


class RequestError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


def _rate_limited(request):
    global _redis_client, _redis_checked
    limit = settings.rate_limit_per_minute()
    if limit <= 0:
        return False
    # Trust the direct peer only. Configure trusted-proxy handling at the ASGI
    # server/load balancer instead of accepting arbitrary client-supplied headers.
    client = request.client.host if request.client else "unknown"
    redis_url = os.environ.get("RATE_LIMIT_REDIS_URL", "").strip()
    if redis_url:
        if not _redis_checked:
            _redis_checked = True
            try:
                from redis import Redis
                _redis_client = Redis.from_url(redis_url, socket_connect_timeout=1, socket_timeout=1,
                                               decode_responses=True)
                _redis_client.ping()
            except Exception:
                logger.exception("Redis rate limiter unavailable; falling back to per-process memory limiter")
                _redis_client = None
        if _redis_client is not None:
            bucket = int(time.time() // 60)
            try:
                count = _redis_client.incr(f"hallucination-detector:rate:{bucket}:{client}")
                if count == 1:
                    _redis_client.expire(f"hallucination-detector:rate:{bucket}:{client}", 65)
                return count > limit
            except Exception:
                logger.exception("Redis rate limiter request failed; using per-process memory limiter")
    now = time.monotonic()
    window = _hits[client]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= limit:
        return True
    window.append(now)
    return False


def _text_option(form, name):
    return str(form.get(name, "") or "").strip()


def _parse_options(form):
    mode = _text_option(form, "mode").lower() or None
    if mode and mode not in settings.VALID_MODES:
        raise RequestError(f"mode must be one of: {', '.join(settings.VALID_MODES)}.")

    policy = _text_option(form, "unsupported_policy").lower() or "keep"
    if policy not in settings.VALID_POLICIES:
        raise RequestError(f"unsupported_policy must be one of: {', '.join(settings.VALID_POLICIES)}.")

    threshold = None
    raw_threshold = _text_option(form, "threshold")
    if raw_threshold:
        try:
            threshold = float(raw_threshold)
        except ValueError:
            raise RequestError("threshold must be a number between 0 and 1.")
        if not 0.0 <= threshold <= 1.0:
            raise RequestError("threshold must be a number between 0 and 1.")

    iterations = None
    raw_iterations = _text_option(form, "max_iterations")
    if raw_iterations:
        if not raw_iterations.isdigit() or int(raw_iterations) > 5:
            raise RequestError("max_iterations must be a whole number from 0 to 5.")
        iterations = int(raw_iterations)

    return {
        "model": _text_option(form, "model") or None,
        "mode": mode,
        "unsupported_policy": policy,
        "threshold": threshold,
        "max_iterations": iterations,
    }


async def _read_text(form, text_field, file_field, label):
    text = str(form.get(text_field, "") or "")
    upload = form.get(file_field)
    if upload is not None and getattr(upload, "filename", ""):
        limit = settings.max_upload_bytes()
        pieces, total = [], 0
        while True:
            piece = await upload.read(min(65536, limit + 1 - total))
            if not piece:
                break
            total += len(piece)
            if total > limit:
                raise RequestError(f"{label} file is larger than {limit} bytes.", 413)
            pieces.append(piece)
        content = b"".join(pieces)
        extracted = read_uploaded_bytes(upload.filename, content)
        if not extracted:
            raise RequestError(f"Could not read any text from the {label.lower()} file. Use a text-based .txt or .pdf.")
        text = extracted
    return text


async def _read_all_sources(form):
    sources = []
    pasted = str(form.get("source_text", "") or "").strip()
    if pasted:
        sources.append({"name": "Pasted source text", "kind": "pasted_text", "text": pasted})

    uploads = form.getlist("source_files") if hasattr(form, "getlist") else []
    # Backward compatibility with the original single-file field.
    if not uploads and form.get("source_file") is not None:
        uploads = [form.get("source_file")]
    total_bytes = 0
    for upload in uploads[:10]:
        if upload is None or not getattr(upload, "filename", ""):
            continue
        pieces = []
        while True:
            piece = await upload.read(65536)
            if not piece:
                break
            total_bytes += len(piece)
            if total_bytes > settings.max_upload_bytes():
                raise RequestError(
                    f"Combined source uploads exceed {settings.max_upload_bytes()} bytes.", 413
                )
            pieces.append(piece)
        extracted = read_uploaded_bytes(upload.filename, b"".join(pieces))
        if not extracted:
            raise RequestError(
                f"Could not read text from {upload.filename}. Supported formats: TXT, MD, PDF, DOCX."
            )
        sources.append({"name": upload.filename, "kind": "uploaded_file", "text": extracted})

    raw_urls = str(form.get("source_urls", "") or "")
    urls = [line.strip() for line in raw_urls.splitlines() if line.strip()]
    if len(urls) > 5:
        raise RequestError("A maximum of 5 HTTPS source URLs is allowed.")
    for url in urls:
        try:
            fetched = await run_in_threadpool(fetch_url_text, url)
        except ValueError as exc:
            raise RequestError(str(exc)) from exc
        sources.append({
            "name": fetched["url"], "kind": "https_url",
            "url": fetched["url"], "text": fetched["text"],
        })

    if not sources:
        raise RequestError("Add pasted source text, upload a document, or provide a trusted HTTPS URL.")
    raw_characters = sum(len(str(item.get("text") or "")) for item in sources)
    if raw_characters > settings.max_source_chars():
        raise RequestError(
            f"Combined source documents exceed {settings.max_source_chars()} characters.", 413
        )
    combined, manifest = combine_sources(sources, max_chars=settings.max_source_chars())
    if not combined:
        raise RequestError("No readable source text was found.")
    return combined, manifest


def _check_lengths(source_text, llm_response):
    if not llm_response.strip():
        raise RequestError("LLM response is empty.")
    if not source_text.strip():
        raise RequestError("Source document is empty.")
    if len(source_text) > settings.max_source_chars():
        raise RequestError(f"Source document is longer than {settings.max_source_chars()} characters.", 413)
    if len(llm_response) > settings.max_response_chars():
        raise RequestError(f"LLM response is longer than {settings.max_response_chars()} characters.", 413)


@app.post("/api/check")
async def api_check(request: Request):
    if _rate_limited(request):
        return JSONResponse(status_code=429, content={"error": "Too many requests. Please wait a minute."})
    try:
        token = settings.api_auth_token()
        if token and request.headers.get("authorization") != f"Bearer {token}":
            return JSONResponse(status_code=401, content={"error": "Authentication required."})
        form = await request.form()
        options = _parse_options(form)
        source_text, source_manifest = await _read_all_sources(form)
        llm_response = await _read_text(form, "llm_response", "response_file", "Response")
        _check_lengths(source_text, llm_response)
        return await run_in_threadpool(run_pipeline, source_text, llm_response, source_manifest=source_manifest, **options)
    except RequestError as error:
        return JSONResponse(status_code=error.status, content={"error": error.message})
    except InvalidLLMOutput as error:
        return JSONResponse(
            status_code=502,
            content={"error": f"The model returned output that could not be parsed: {error}"},
        )
    except Exception:
        logger.exception("Unhandled verification request error")
        return JSONResponse(status_code=500, content={"error": "Internal server error. Check server logs for details."})


@app.get("/api/config")
async def api_config():
    return {
        "verifier_mode": settings.verifier_mode(),
        "nli_available": is_available(),
        "calibrated": calibration.get_calibrator() is not None,
        "cascade_threshold": settings.cascade_threshold(),
        "max_iterations": settings.max_iterations(),
        "max_source_chars": settings.max_source_chars(),
        "max_response_chars": settings.max_response_chars(),
        "rag_enabled": True,
        "rag_top_k": settings.rag_top_k(),
        "rag_context_chars": settings.rag_context_chars(),
        "multi_source_ingestion": True,
        "supported_source_types": ["pasted_text", "txt", "md", "pdf", "docx", "public_https_html", "public_https_text"],
        "auth_required": bool(settings.api_auth_token()),
    }


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
