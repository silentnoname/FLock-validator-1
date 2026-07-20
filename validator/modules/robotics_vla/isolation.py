from __future__ import annotations

import base64
import ctypes
import json
import os
import platform
import select
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from validator.modules.robotics_vla.errors import RoboticsSubmissionError
from validator.modules.robotics_vla.memory_monitor import MemoryMonitor


_HEADER = struct.Struct("!Q")
_MAX_REQUEST_BYTES = 32 * 1024**2
_MAX_RESPONSE_BYTES = 1024**2
_MAX_STDERR_CAPTURE_BYTES = 64 * 1024
_MAX_STDERR_REPORT_BYTES = 4 * 1024
_MAX_WORKER_TRACEBACK_CHARS = 8 * 1024
_SANDBOX_ENV_ALLOWLIST = {
    "CUDA_VISIBLE_DEVICES",
    "DYLD_LIBRARY_PATH",
    "LD_LIBRARY_PATH",
    "MKL_NUM_THREADS",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "PATH",
    "TOKENIZERS_PARALLELISM",
}


class _BoundedStderrCapture:
    """Continuously drain worker stderr while retaining only a bounded tail."""

    def __init__(self, stream: Any, max_bytes: int = _MAX_STDERR_CAPTURE_BYTES) -> None:
        self._stream = stream
        self._max_bytes = max_bytes
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._drain,
            name="robotics-vla-worker-stderr",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def wait(self, timeout: float = 0.2) -> None:
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def close(self) -> None:
        self.wait()
        try:
            self._stream.close()
        except OSError:
            pass

    def tail(self, max_bytes: int = _MAX_STDERR_REPORT_BYTES) -> str:
        with self._lock:
            data = bytes(self._buffer[-max_bytes:])
        return data.decode("utf-8", errors="replace").strip()

    def _drain(self) -> None:
        try:
            fd = self._stream.fileno()
            while True:
                chunk = os.read(fd, 4096)
                if not chunk:
                    return
                with self._lock:
                    self._buffer.extend(chunk)
                    overflow = len(self._buffer) - self._max_bytes
                    if overflow > 0:
                        del self._buffer[:overflow]
        except (OSError, ValueError):
            return


class IsolatedPolicy:
    """Safe proxy for a miner policy running in a separate sandbox process."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        temp_dir: tempfile.TemporaryDirectory[str],
        action_timeout_seconds: float,
        audited_parameter_count: int | None,
        stderr_capture: _BoundedStderrCapture,
        memory_monitor: MemoryMonitor | None = None,
    ) -> None:
        self._process = process
        self._temp_dir = temp_dir
        self._action_timeout_seconds = action_timeout_seconds
        # Telemetry only (int, or None when the graph could not be fully walked).
        # Model size is enforced by the memory monitor below, not this count.
        self.audited_parameter_count = audited_parameter_count
        self._stderr_capture = stderr_capture
        self._memory_monitor = memory_monitor
        self._closed = False

    @classmethod
    def start(
        cls,
        *,
        model_dir: Path,
        adapter_filename: str,
        device: str,
        torch_dtype: str,
        load_timeout_seconds: float,
        action_timeout_seconds: float,
        memory_limit_bytes: int,
        cpu_time_seconds: int,
    ) -> "IsolatedPolicy":
        if (
            min(
                load_timeout_seconds,
                action_timeout_seconds,
                memory_limit_bytes,
                cpu_time_seconds,
            )
            <= 0
        ):
            raise ValueError("Policy sandbox limits must all be positive")

        _protect_parent_secrets()
        temp_dir = tempfile.TemporaryDirectory(prefix="robotics-vla-worker-")
        try:
            command = _worker_command(
                model_dir=model_dir,
                adapter_filename=adapter_filename,
                device=device,
                torch_dtype=torch_dtype,
                memory_limit_bytes=memory_limit_bytes,
                cpu_time_seconds=cpu_time_seconds,
            )
            env = _sandbox_environment(Path(temp_dir.name))
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=temp_dir.name,
                env=env,
                close_fds=True,
                start_new_session=True,
            )
        except Exception:
            temp_dir.cleanup()
            raise

        if process.stderr is None:
            _kill_process_group(process)
            temp_dir.cleanup()
            raise RuntimeError("Policy sandbox stderr pipe is unavailable")
        stderr_capture = _BoundedStderrCapture(process.stderr)
        stderr_capture.start()

        # Enforce the model-size ceiling on the live process: however the weights
        # are represented, they occupy memory. This is the authoritative bound;
        # the parameter count reported below is telemetry only.
        monitor = MemoryMonitor(process.pid, memory_limit_bytes)
        monitor.start()
        proxy = cls(
            process,
            temp_dir,
            action_timeout_seconds,
            audited_parameter_count=None,
            stderr_capture=stderr_capture,
            memory_monitor=monitor,
        )
        try:
            response = proxy._receive(load_timeout_seconds, "model_load_timeout")
            proxy._raise_if_error(response, "model_load_failed")
            parameter_count = response.get("parameter_count")
            proxy.audited_parameter_count = (
                parameter_count
                if isinstance(parameter_count, int) and parameter_count >= 0
                else None
            )
            return proxy
        except Exception:
            proxy.close()
            raise

    def act(self, obs: dict[str, Any]) -> Any:
        if self._closed:
            raise RoboticsSubmissionError(
                "Policy sandbox is no longer running",
                failure_mode="policy_execution_failed",
            )
        self._send({"op": "act", "obs": _encode_value(obs)})
        response = self._receive(self._action_timeout_seconds, "policy_timeout")
        self._raise_if_error(response, "policy_execution_failed")
        return np.asarray(response.get("action"))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._memory_monitor is not None:
            self._memory_monitor.stop()
        process = self._process
        try:
            if process.poll() is None:
                try:
                    self._send({"op": "close"})
                    process.wait(timeout=1)
                except Exception:
                    _kill_process_group(process)
        finally:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            self._stderr_capture.close()
            try:
                self._temp_dir.cleanup()
            except OSError:
                # The worker is already dead; cleanup failure must not replace a
                # validation result or the original sandbox error.
                pass

    def __enter__(self) -> "IsolatedPolicy":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _send(self, message: dict[str, Any]) -> None:
        if self._process.stdin is None or self._process.poll() is not None:
            # If the monitor killed the worker for exceeding its memory budget,
            # report that rather than a generic exit — otherwise the per-action
            # retry loop would mask the real cause on subsequent sends.
            self._raise_if_memory_exceeded()
            message = "Policy sandbox exited unexpectedly"
            raise RoboticsSubmissionError(
                self._with_exit_diagnostics(message),
                failure_mode="policy_execution_failed",
                submission_message=message,
            )
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
        if len(payload) > _MAX_REQUEST_BYTES:
            raise RoboticsSubmissionError(
                "Policy observation exceeds the sandbox message limit",
                failure_mode="policy_protocol_error",
            )
        try:
            self._process.stdin.write(_HEADER.pack(len(payload)))
            self._process.stdin.write(payload)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._wait_for_worker_exit()
            self._raise_if_memory_exceeded(exc)
            message = "Policy sandbox exited while receiving an observation"
            raise RoboticsSubmissionError(
                self._with_exit_diagnostics(message),
                failure_mode="policy_execution_failed",
                submission_message=message,
            ) from exc

    def _receive(self, timeout_seconds: float, timeout_mode: str) -> dict[str, Any]:
        if self._process.stdout is None:
            raise RoboticsSubmissionError(
                "Policy sandbox protocol is unavailable",
                failure_mode="policy_protocol_error",
            )
        fd = self._process.stdout.fileno()
        deadline = time.monotonic() + timeout_seconds
        try:
            header = _read_exact(fd, _HEADER.size, deadline)
            size = _HEADER.unpack(header)[0]
            if size > _MAX_RESPONSE_BYTES:
                raise RoboticsSubmissionError(
                    "Policy sandbox returned an oversized response",
                    failure_mode="policy_protocol_error",
                )
            payload = _read_exact(fd, size, deadline)
            response = json.loads(payload.decode("utf-8"))
            if not isinstance(response, dict):
                raise ValueError("response must be a JSON object")
            return response
        except TimeoutError as exc:
            _kill_process_group(self._process)
            self._raise_if_memory_exceeded(exc)
            raise RoboticsSubmissionError(
                f"Policy sandbox exceeded its {timeout_seconds:g}s wall-time limit",
                failure_mode=timeout_mode,
            ) from exc
        except RoboticsSubmissionError:
            _kill_process_group(self._process)
            raise
        except (
            EOFError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            self._wait_for_worker_exit()
            diagnostics = self._exit_diagnostics()
            _kill_process_group(self._process)
            # A memory-limit kill closes the pipe; surface it as such rather than a
            # generic protocol error so the miner sees why the submission failed.
            self._raise_if_memory_exceeded(exc)
            submission_message = (
                f"Policy sandbox returned an invalid protocol response: {exc}"
            )
            message = submission_message
            if diagnostics:
                message = f"{message} ({diagnostics})"
            raise RoboticsSubmissionError(
                message,
                failure_mode="policy_protocol_error",
                submission_message=submission_message,
            ) from exc

    def _wait_for_worker_exit(self, timeout: float = 0.1) -> None:
        if self._process.poll() is not None:
            return
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass

    def _with_exit_diagnostics(self, message: str) -> str:
        diagnostics = self._exit_diagnostics()
        return f"{message} ({diagnostics})" if diagnostics else message

    def _exit_diagnostics(self) -> str:
        returncode = self._process.poll()
        if returncode is not None:
            self._stderr_capture.wait()

        details: list[str] = []
        if returncode is not None:
            if returncode < 0:
                signal_number = -returncode
                try:
                    signal_name = signal.Signals(signal_number).name
                except ValueError:
                    signal_name = "unknown signal"
                details.append(f"signal {signal_name} ({signal_number})")
            else:
                details.append(f"exit status {returncode}")

        stderr_tail = self._stderr_capture.tail()
        if stderr_tail:
            details.append(f"worker stderr tail:\n{stderr_tail}")
        return "; ".join(details)

    def _raise_if_memory_exceeded(self, cause: BaseException | None = None) -> None:
        monitor = self._memory_monitor
        if monitor is None or not monitor.breached:
            return
        limit_gib = monitor.limit_bytes / 1024**3
        observed_gib = monitor.observed_at_kill / 1024**3
        raise RoboticsSubmissionError(
            f"Policy sandbox exceeded its {limit_gib:.0f} GiB memory limit "
            f"(observed {observed_gib:.1f} GiB)",
            failure_mode="policy_memory_exceeded",
        ) from cause

    @staticmethod
    def _raise_if_error(response: dict[str, Any], fallback_mode: str) -> None:
        if response.get("ok") is True:
            return
        message = response.get("error")
        fallback_message = (
            str(message) if message else "Policy sandbox rejected the request"
        )
        submitted_message = response.get("submission_error")
        submission_message = (
            str(submitted_message) if submitted_message else fallback_message
        )
        message = fallback_message
        worker_traceback = response.get("traceback")
        if isinstance(worker_traceback, str) and worker_traceback.strip():
            traceback_tail = worker_traceback[-_MAX_WORKER_TRACEBACK_CHARS:].strip()
            message = f"{message}\nWorker traceback (tail):\n{traceback_tail}"
        failure_mode = response.get("failure_mode")
        raise RoboticsSubmissionError(
            message,
            failure_mode=str(failure_mode) if failure_mode else fallback_mode,
            submission_message=submission_message,
        )


def _worker_command(
    *,
    model_dir: Path,
    adapter_filename: str,
    device: str,
    torch_dtype: str,
    memory_limit_bytes: int,
    cpu_time_seconds: int,
) -> list[str]:
    worker = Path(__file__).with_name("adapter_worker.py")
    base = [
        sys.executable,
        "-u",
        str(worker),
        "--model-dir",
        str(model_dir),
        "--adapter-filename",
        adapter_filename,
        "--device",
        device,
        "--torch-dtype",
        torch_dtype,
        "--memory-limit-bytes",
        str(memory_limit_bytes),
        "--cpu-time-seconds",
        str(cpu_time_seconds),
    ]
    system = platform.system()
    if system == "Linux":
        return base
    if system == "Darwin":
        # Tests use trusted fixture adapters and may themselves run inside a host
        # sandbox that forbids nesting sandbox-exec. The worker still has a clean
        # environment, a separate process, and resource/time limits in this mode.
        if (
            os.getenv("PYTEST_CURRENT_TEST")
            or os.getenv("ROBOTICS_VLA_ALLOW_UNSAFE_LOCAL_ADAPTER") == "1"
        ):
            return base
        if shutil.which("sandbox-exec"):
            profile = "(version 1)(allow default)(deny network*)"
            return ["sandbox-exec", "-p", profile, *base]
    raise RoboticsSubmissionError(
        f"No supported network sandbox is available on {system or 'this platform'}",
        failure_mode="sandbox_unavailable",
    )


def _sandbox_environment(temp_dir: Path) -> dict[str, str]:
    env = {
        key: value for key, value in os.environ.items() if key in _SANDBOX_ENV_ALLOWLIST
    }
    env.update(
        {
            "HOME": str(temp_dir),
            "TMPDIR": str(temp_dir),
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def _protect_parent_secrets() -> None:
    """Prevent same-UID workers from reading the parent through /proc on Linux."""
    if platform.system() != "Linux":
        return
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(4, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_DUMPABLE) failed")
    except (AttributeError, OSError) as exc:
        raise RoboticsSubmissionError(
            f"Could not protect validator credentials from the policy worker: {exc}",
            failure_mode="sandbox_unavailable",
        ) from exc


def _encode_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": base64.b64encode(array.tobytes()).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise RoboticsSubmissionError(
                "Policy observations may only contain string-keyed dictionaries",
                failure_mode="policy_protocol_error",
            )
        return {key: _encode_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise RoboticsSubmissionError(
        f"Policy observation contains unsupported value {type(value).__name__}",
        failure_mode="policy_protocol_error",
    )


def decode_value(value: Any) -> Any:
    """Decode a trusted parent request inside the worker."""
    if isinstance(value, dict) and value.get("__ndarray__") is True:
        dtype = np.dtype(value["dtype"])
        shape = tuple(int(dim) for dim in value["shape"])
        raw = base64.b64decode(value["data"], validate=True)
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if expected != len(raw):
            raise ValueError("ndarray payload size does not match its shape")
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if isinstance(value, dict):
        return {key: decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_value(item) for item in value]
    return value


def read_worker_message(stream: Any) -> dict[str, Any]:
    header = _read_blocking(stream, _HEADER.size)
    size = _HEADER.unpack(header)[0]
    if size > _MAX_REQUEST_BYTES:
        raise ValueError("request exceeds worker message limit")
    payload = _read_blocking(stream, size)
    message = json.loads(payload.decode("utf-8"))
    if not isinstance(message, dict):
        raise ValueError("request must be a JSON object")
    return message


def write_worker_message(stream: Any, message: dict[str, Any]) -> None:
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    if len(payload) > _MAX_RESPONSE_BYTES:
        message = {
            "ok": False,
            "error": "Policy response exceeds the sandbox message limit",
            "failure_mode": "policy_protocol_error",
        }
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    stream.write(_HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def _read_exact(fd: int, size: int, deadline: float) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        readable, _, _ = select.select([fd], [], [], remaining)
        if not readable:
            raise TimeoutError
        chunk = os.read(fd, size - len(chunks))
        if not chunk:
            raise EOFError("policy sandbox closed the protocol pipe")
        chunks.extend(chunk)
    return bytes(chunks)


def _read_blocking(stream: Any, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError("validator closed the protocol pipe")
        chunks.extend(chunk)
    return bytes(chunks)


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)
