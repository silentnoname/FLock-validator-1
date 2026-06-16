from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlparse

from validator.exceptions import LLMJudgeException

FLOCK_API_HOST = "api.flock.io"
FLOCK_API_EVAL_MODEL_ALIASES = {
    "gemini-3.5-flash": "gemini-3.5-flash-flocklife",
    "deepseek-v4-pro": "deepseek-v4-pro-dslife",
    "deepseek-v4-flash": "deepseek-v4-flash-dsikh",
    "kimi-k2.6": "kimi-k2.6-llm",
    "gemini-3.1-pro": "gemini-3.1-pro-deai",
}
FLOCK_API_EVAL_MODELS = list(FLOCK_API_EVAL_MODEL_ALIASES.values())


def _is_flock_api_platform(openai_base_url: str | None) -> bool:
    try:
        return urlparse(openai_base_url or "").netloc == FLOCK_API_HOST
    except Exception:
        return False


def _map_flock_eval_model(model_name: str) -> str:
    if model_name in FLOCK_API_EVAL_MODEL_ALIASES.values():
        return model_name

    _, _, body = model_name.rpartition("/")
    parts = body.split("-")
    alias_tokens = []
    while parts and parts[-1] in {"low", "high", "thinking"}:
        alias_tokens.insert(0, parts.pop())

    base_body = "-".join(parts)
    mapped_body = FLOCK_API_EVAL_MODEL_ALIASES.get(base_body)
    if mapped_body is None:
        mapped_body = FLOCK_API_EVAL_MODEL_ALIASES.get(body)

    if mapped_body is None:
        return model_name

    if alias_tokens:
        return mapped_body + "-" + "-".join(alias_tokens)
    return mapped_body


def resolve_eval_models(
    eval_args: Mapping[str, Any] | None,
    available_models: Iterable[str],
    openai_base_url: str | None = None,
) -> list[str]:
    requested_models = eval_args.get("eval_model_list", []) if eval_args else []
    is_flock_api = _is_flock_api_platform(openai_base_url)

    if not isinstance(requested_models, list):
        raise LLMJudgeException("eval_model_list must be a list of model names")

    if requested_models:
        if not all(
            isinstance(model, str) and model.strip() for model in requested_models
        ):
            raise LLMJudgeException(
                "eval_model_list must contain only non-empty model names"
            )
        if is_flock_api:
            requested_models = [
                _map_flock_eval_model(model_name) for model_name in requested_models
            ]
        return list(dict.fromkeys(requested_models))

    if is_flock_api:
        return FLOCK_API_EVAL_MODELS

    discovered_models = [
        model
        for model in available_models
        if isinstance(model, str) and model.strip()
    ]
    discovered_models = list(dict.fromkeys(discovered_models))
    if discovered_models:
        return discovered_models

    raise LLMJudgeException(
        "No evaluation models configured and provider model discovery returned none. "
        "Set eval_args.eval_model_list explicitly."
    )
