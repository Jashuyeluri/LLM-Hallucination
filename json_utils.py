import json

import llm_client


class InvalidLLMOutput(ValueError):
    pass


def strip_fence(raw):
    text = raw.strip()
    if text.startswith("```"):
        newline = text.find("\n")
        text = text[newline + 1:] if newline >= 0 else ""
        text = text.rstrip()
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def extract_json_array(raw):
    text = strip_fence(raw)
    decoder = json.JSONDecoder()
    best = None
    best_span = -1
    for index, char in enumerate(text):
        if char != "[":
            continue
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and end - index > best_span:
            best, best_span = value, end - index
    if best is None:
        raise InvalidLLMOutput("No valid JSON array found in the model output.")
    return best


def chat_json_array(prompt, validate, model=None, retries=2):
    attempt_prompt = prompt
    last_error = InvalidLLMOutput("The model produced no usable output.")
    for _ in range(retries + 1):
        raw = llm_client.chat(attempt_prompt, model=model)
        try:
            return validate(extract_json_array(raw))
        except InvalidLLMOutput as error:
            last_error = error
            attempt_prompt = (
                f"{prompt}\n\nYour previous reply was rejected: {error} "
                "Reply again with ONLY the JSON array."
            )
    raise last_error
