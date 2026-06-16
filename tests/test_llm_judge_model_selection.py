import pytest

from validator.exceptions import LLMJudgeException
from validator.modules.llm_judge_model_selection import (
    resolve_eval_models,
    resolve_eval_temperature,
)


def test_configured_models_are_authoritative():
    models = resolve_eval_models(
        {"eval_model_list": ["kimi-k2.6-thinking", "qwen3.5", "qwen3.5"]},
        ["gpt-4o"],
    )

    assert models == ["kimi-k2.6-thinking", "qwen3.5"]


def test_configured_models_work_when_discovery_fails():
    models = resolve_eval_models(
        {"eval_model_list": ["kimi-k2.6"]},
        [],
    )

    assert models == ["kimi-k2.6"]


def test_flock_api_maps_configured_model_aliases():
    models = resolve_eval_models(
        {"eval_model_list": ["kimi-k2.6-thinking", "gemini-3.1-pro", "gemini-3.1-pro"]},
        [],
        openai_base_url="https://api.flock.io/v1",
    )

    assert models == ["kimi-k2.6-llm-thinking", "gemini-3.1-pro-deai"]


def test_flock_api_uses_flock_models_when_config_is_absent():
    models = resolve_eval_models(
        {},
        [],
        openai_base_url="https://api.flock.io/v1",
    )

    assert models == [
        "gemini-3.5-flash-flocklife",
        "deepseek-v4-pro-dslife",
        "deepseek-v4-flash-dsikh",
        "kimi-k2.6-llm",
        "gemini-3.1-pro-deai",
    ]


@pytest.mark.parametrize(
    "eval_model,expected_temperature",
    [
        ("kimi-k2.6", 0.6),
        ("kimi-k2.6-llm", 1),
        ("moonshotai/kimi-k2.6-llm", 1),
        ("kimi-k2.6-thinking", 1),
        ("kimi-k2.6-llm-thinking", 1),
        ("moonshotai/kimi-k2.6-llm-thinking", 1),
    ],
)
def test_kimi_eval_temperature_overrides(eval_model, expected_temperature):
    assert resolve_eval_temperature(eval_model, 0.1) == expected_temperature


def test_non_kimi_eval_temperature_uses_configured_default():
    assert resolve_eval_temperature("gemini-3.1-pro-deai", 0.2) == 0.2


def test_discovered_models_are_used_when_config_is_absent():
    models = resolve_eval_models({}, ["model-a", "model-b", "model-a"])

    assert models == ["model-a", "model-b"]


def test_missing_config_and_discovery_fails_explicitly():
    with pytest.raises(LLMJudgeException, match="eval_model_list"):
        resolve_eval_models({}, [])


@pytest.mark.parametrize(
    "configured_models",
    ["model-a", "", None, ["model-a", ""], ["model-a", None]],
)
def test_invalid_configured_models_fail_explicitly(configured_models):
    with pytest.raises(LLMJudgeException, match="eval_model_list"):
        resolve_eval_models({"eval_model_list": configured_models}, ["gpt-4o"])
