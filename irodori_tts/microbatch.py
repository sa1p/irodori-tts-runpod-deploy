from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Generic, Hashable, TypeVar


PayloadT = TypeVar("PayloadT")
ResultT = TypeVar("ResultT")


class MicrobatchQueueFull(RuntimeError):
    pass


class MicrobatchClosed(RuntimeError):
    pass


@dataclass(frozen=True)
class MicrobatchResult(Generic[ResultT]):
    value: ResultT
    batch_size: int
    queue_wait_seconds: float
    batch_seconds: float


@dataclass
class _Pending(Generic[PayloadT, ResultT]):
    group_key: Hashable
    payload: PayloadT
    submitted_at: float
    done: threading.Event
    result: ResultT | None = None
    error: BaseException | None = None
    batch_size: int = 0
    queue_wait_seconds: float = 0.0
    batch_seconds: float = 0.0
    cancelled: bool = False


class MicrobatchExecutor(Generic[PayloadT, ResultT]):
    """Collect compatible items briefly and execute them on one background thread."""

    def __init__(
        self,
        execute: Callable[[list[PayloadT]], list[ResultT]],
        *,
        max_batch_size: int,
        window_ms: float,
        max_queue_size: int,
        thread_name: str = "microbatch-executor",
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if window_ms < 0:
            raise ValueError("window_ms must be non-negative")
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")

        self._execute = execute
        self._max_batch_size = int(max_batch_size)
        self._window_seconds = float(window_ms) / 1000.0
        self._max_queue_size = int(max_queue_size)
        self._condition = threading.Condition()
        self._pending: list[_Pending[PayloadT, ResultT]] = []
        self._closed = False
        self._inflight = 0
        self._batches_total = 0
        self._items_total = 0
        self._max_observed_batch_size = 0
        self._last_batch_size = 0
        self._last_batch_seconds = 0.0
        self._thread = threading.Thread(target=self._run, name=thread_name, daemon=True)
        self._thread.start()

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    @property
    def window_ms(self) -> float:
        return self._window_seconds * 1000.0

    def stats(self) -> dict[str, int | float | bool]:
        with self._condition:
            return {
                "closed": self._closed,
                "pending": len([item for item in self._pending if not item.cancelled]),
                "inflight": self._inflight,
                "batches_total": self._batches_total,
                "items_total": self._items_total,
                "max_observed_batch_size": self._max_observed_batch_size,
                "max_batch_size": self._max_batch_size,
                "max_queue_size": self._max_queue_size,
                "window_ms": self.window_ms,
                "last_batch_size": self._last_batch_size,
                "last_batch_seconds": self._last_batch_seconds,
            }

    def submit(
        self,
        group_key: Hashable,
        payload: PayloadT,
        *,
        timeout: float | None = None,
    ) -> MicrobatchResult[ResultT]:
        item = _Pending[PayloadT, ResultT](
            group_key=group_key,
            payload=payload,
            submitted_at=time.perf_counter(),
            done=threading.Event(),
        )
        with self._condition:
            if self._closed:
                raise MicrobatchClosed("microbatch executor is closed")
            pending_count = len([pending for pending in self._pending if not pending.cancelled])
            if pending_count >= self._max_queue_size:
                raise MicrobatchQueueFull("microbatch queue is full")
            self._pending.append(item)
            self._condition.notify_all()

        completed = item.done.wait(timeout=timeout)
        if not completed:
            with self._condition:
                item.cancelled = True
                self._condition.notify_all()
            raise TimeoutError("microbatch result timed out")
        if item.error is not None:
            raise item.error
        return MicrobatchResult(
            value=item.result,  # type: ignore[arg-type]
            batch_size=item.batch_size,
            queue_wait_seconds=item.queue_wait_seconds,
            batch_seconds=item.batch_seconds,
        )

    def close(self, *, timeout: float = 2.0) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            error = MicrobatchClosed("microbatch executor was closed")
            for item in self._pending:
                if item.cancelled:
                    continue
                item.error = error
                item.done.set()
            self._pending.clear()
            self._condition.notify_all()
        self._thread.join(timeout=max(0.0, timeout))

    def _pop_next_batch(self) -> list[_Pending[PayloadT, ResultT]] | None:
        with self._condition:
            while True:
                self._pending = [item for item in self._pending if not item.cancelled]
                if self._pending:
                    break
                if self._closed:
                    return None
                self._condition.wait()

            first = self._pending.pop(0)
            batch = [first]
            deadline = first.submitted_at + self._window_seconds
            while len(batch) < self._max_batch_size:
                compatible_index = next(
                    (
                        index
                        for index, item in enumerate(self._pending)
                        if not item.cancelled and item.group_key == first.group_key
                    ),
                    None,
                )
                if compatible_index is not None:
                    batch.append(self._pending.pop(compatible_index))
                    continue

                remaining = deadline - time.perf_counter()
                if remaining <= 0 or self._closed:
                    break
                self._condition.wait(timeout=remaining)
                self._pending = [item for item in self._pending if not item.cancelled]
            return batch

    def _run(self) -> None:
        while True:
            batch = self._pop_next_batch()
            if batch is None:
                return
            active = [item for item in batch if not item.cancelled]
            if not active:
                continue

            started_at = time.perf_counter()
            with self._condition:
                self._inflight = len(active)
            try:
                values = list(self._execute([item.payload for item in active]))
                if len(values) != len(active):
                    raise RuntimeError(
                        "microbatch executor returned "
                        f"{len(values)} results for {len(active)} payloads"
                    )
            except BaseException as exc:
                elapsed = time.perf_counter() - started_at
                for item in active:
                    if item.cancelled:
                        continue
                    item.error = exc
                    item.batch_size = len(active)
                    item.queue_wait_seconds = started_at - item.submitted_at
                    item.batch_seconds = elapsed
                    item.done.set()
            else:
                elapsed = time.perf_counter() - started_at
                for item, value in zip(active, values, strict=True):
                    if item.cancelled:
                        continue
                    item.result = value
                    item.batch_size = len(active)
                    item.queue_wait_seconds = started_at - item.submitted_at
                    item.batch_seconds = elapsed
                    item.done.set()
            finally:
                with self._condition:
                    self._inflight = 0
                    self._batches_total += 1
                    self._items_total += len(active)
                    self._max_observed_batch_size = max(
                        self._max_observed_batch_size,
                        len(active),
                    )
                    self._last_batch_size = len(active)
                    self._last_batch_seconds = time.perf_counter() - started_at
                    self._condition.notify_all()
