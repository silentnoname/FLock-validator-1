from __future__ import annotations


class RoboticsSubmissionError(Exception):
    """Raised when a miner's submission is invalid or unrunnable.

    These are the *submitter's* fault (a broken adapter, an unloadable model, a
    policy that emits malformed actions, or a model that violates the parameter
    cap). They must be scored as an invalid submission (score 0) and must NOT
    crash the long-running validator. Genuine infrastructure problems (a broken
    validation package, a simulator crash, network/disk failures) deliberately do
    NOT use this type, so they can propagate to the runner and be retried or have
    the assignment re-queued instead of unfairly zeroing the miner.

    ``failure_mode`` is a short, stable, machine-readable tag surfaced in the
    submitted metrics' diagnostics so failures can be triaged.
    """

    def __init__(
        self,
        message: str,
        failure_mode: str = "submission_error",
        *,
        submission_message: str | None = None,
    ):
        super().__init__(message)
        self.failure_mode = failure_mode
        # Full worker reports stay in local logs. The submitted metrics retain the
        # original concise diagnostics contract expected by FedLedger.
        self.submission_message = (
            message if submission_message is None else submission_message
        )
