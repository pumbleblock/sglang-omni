# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading

from sglang_omni_v1.scheduling.messages import IncomingMessage
from sglang_omni_v1.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler
from tests.unit_test.pipeline.helpers import run_scheduler


def test_threaded_simple_scheduler_runs_requests_concurrently() -> None:
    started: list[str] = []
    lock = threading.Lock()
    both_started = threading.Event()
    release = threading.Event()

    def compute(payload: str) -> str:
        with lock:
            started.append(payload)
            if len(started) == 2:
                both_started.set()
        assert release.wait(timeout=2.0)
        return payload

    scheduler = ThreadedSimpleScheduler(compute, max_concurrency=2)
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    try:
        scheduler.inbox.put(IncomingMessage("req-1", "new_request", "one"))
        scheduler.inbox.put(IncomingMessage("req-2", "new_request", "two"))

        assert both_started.wait(timeout=2.0)
        release.set()

        outputs = [scheduler.outbox.get(timeout=2.0) for _ in range(2)]
        assert {output.request_id for output in outputs} == {"req-1", "req-2"}
        assert {output.data for output in outputs} == {"one", "two"}
    finally:
        release.set()
        scheduler.stop()
        thread.join(timeout=2.0)


def test_threaded_simple_scheduler_reports_worker_errors() -> None:
    def compute(payload: str) -> str:
        raise RuntimeError(payload)

    outputs = run_scheduler(
        ThreadedSimpleScheduler(compute, max_concurrency=1),
        [IncomingMessage("req-err", "new_request", "boom")],
        output_count=1,
    )

    assert outputs[0].request_id == "req-err"
    assert outputs[0].type == "error"
    assert isinstance(outputs[0].data, RuntimeError)
