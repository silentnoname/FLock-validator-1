import os

from validator import conda


def test_run_command_flushes_forwarded_output(monkeypatch):
    printed = []

    class FakeProcess:
        stdout = ["ready\n"]
        returncode = 0

        def wait(self):
            return self.returncode

    monkeypatch.setattr(conda.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(
        "builtins.print",
        lambda value, **kwargs: printed.append((value, kwargs)),
    )

    conda.run_command(["worker"])

    assert printed == [("ready\n", {"end": "", "flush": True})]


def test_run_in_env_preserves_environment_and_forces_unbuffered_output(monkeypatch):
    calls = []
    monkeypatch.setenv("PATH", "/test/path")
    monkeypatch.setattr(
        conda,
        "run_command",
        lambda cmd, **kwargs: calls.append((cmd, kwargs)),
    )

    conda.run_in_env(
        "test-env",
        ["--no-capture-output", "python", "worker.py"],
        {"CUSTOM_SETTING": "enabled", "PYTHONUNBUFFERED": "0"},
    )

    command, kwargs = calls[0]
    assert command == [
        "conda",
        "run",
        "-n",
        "test-env",
        "--no-capture-output",
        "python",
        "worker.py",
    ]
    assert kwargs["env"]["PATH"] == "/test/path"
    assert kwargs["env"]["CUSTOM_SETTING"] == "enabled"
    assert kwargs["env"]["PYTHONUNBUFFERED"] == "1"


def test_unbuffered_env_does_not_mutate_process_environment(monkeypatch):
    monkeypatch.delenv("PYTHONUNBUFFERED", raising=False)

    merged_env = conda._unbuffered_env({"CUSTOM_SETTING": "enabled"})

    assert merged_env["CUSTOM_SETTING"] == "enabled"
    assert merged_env["PYTHONUNBUFFERED"] == "1"
    assert "PYTHONUNBUFFERED" not in os.environ
