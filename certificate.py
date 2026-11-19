import hashlib
import hmac
import json
import os
import time

SCHEMA = "hallucination-verification-certificate/1"


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _record(result):
    return {
        "claim": result["claim"],
        "label": result["label"],
        "confidence": result["confidence"],
        "verifier": result.get("verifier", "llm"),
        "guard_flags": result.get("guard_flags", []),
        "evidence": [
            {"start": item.get("start"), "end": item.get("end"), "sha256": _sha(item.get("text", ""))}
            for item in result.get("evidence", [])
        ],
    }


def _genesis(source_sha, response_sha, config):
    return _sha(source_sha + response_sha + _canonical(config))


def _chain(genesis, records):
    previous = genesis
    for record in records:
        body = {key: value for key, value in record.items() if key != "chain_hash"}
        record["chain_hash"] = _sha(previous + _canonical(body))
        previous = record["chain_hash"]
    return previous


def _sign(root_hash, key):
    return hmac.new(key.encode("utf-8"), root_hash.encode("utf-8"), hashlib.sha256).hexdigest()


def build_certificate(source_text, llm_response, results, config, corrected_response=None, signing_key=None):
    source_sha, response_sha = _sha(source_text), _sha(llm_response)
    records = [_record(result) for result in results]
    root = _chain(_genesis(source_sha, response_sha, config), records)
    key = signing_key if signing_key is not None else os.environ.get("CERT_SIGNING_KEY", "")
    return {
        "schema": SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_sha256": source_sha,
        "response_sha256": response_sha,
        "corrected_sha256": _sha(corrected_response) if corrected_response else None,
        "config": config,
        "records": records,
        "root_hash": root,
        "signature": _sign(root, key) if key else None,
    }


def verify_certificate(certificate, source_text, signing_key=None):
    errors = []
    if certificate.get("schema") != SCHEMA:
        errors.append("unknown_schema")
    if certificate.get("source_sha256") != _sha(source_text):
        errors.append("source_hash_mismatch")
    records = [dict(record) for record in certificate.get("records", [])]
    claimed = [record.get("chain_hash") for record in records]
    genesis = _genesis(certificate.get("source_sha256", ""), certificate.get("response_sha256", ""), certificate.get("config", {}))
    root = _chain(genesis, records)
    if claimed != [record["chain_hash"] for record in records] or root != certificate.get("root_hash"):
        errors.append("chain_broken")
    for index, record in enumerate(records):
        evidence = record.get("evidence", [])
        if record.get("label") in ("supported", "contradicted") and not evidence:
            errors.append(f"record_{index}_uncited_verdict")
        for span in evidence:
            start, end = span.get("start"), span.get("end")
            valid = isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(source_text)
            if not valid or _sha(source_text[start:end]) != span.get("sha256"):
                errors.append(f"record_{index}_evidence_mismatch")
    key = signing_key if signing_key is not None else os.environ.get("CERT_SIGNING_KEY", "")
    if certificate.get("signature"):
        if not key:
            errors.append("signature_present_but_no_key")
        elif not hmac.compare_digest(certificate["signature"], _sign(certificate.get("root_hash", ""), key)):
            errors.append("signature_invalid")
    return {"valid": not errors, "errors": errors, "root_hash": certificate.get("root_hash")}
