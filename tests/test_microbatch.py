from __future__ import annotations

import threading
import time
import unittest

from irodori_tts.microbatch import MicrobatchExecutor


class MicrobatchExecutorTest(unittest.TestCase):
    def test_collects_compatible_requests_and_preserves_results(self) -> None:
        observed: list[list[int]] = []

        def execute(values: list[int]) -> list[int]:
            observed.append(list(values))
            return [value * 10 for value in values]

        executor = MicrobatchExecutor(
            execute,
            max_batch_size=4,
            window_ms=50,
            max_queue_size=16,
        )
        barrier = threading.Barrier(5)
        results: dict[int, tuple[int, int]] = {}

        def submit(value: int) -> None:
            barrier.wait()
            result = executor.submit("same", value, timeout=2.0)
            results[value] = (result.value, result.batch_size)

        threads = [threading.Thread(target=submit, args=(value,)) for value in range(4)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=2.0)
        executor.close()

        self.assertEqual(sorted(observed[0]), [0, 1, 2, 3])
        self.assertEqual(
            {key: value[0] for key, value in results.items()},
            {0: 0, 1: 10, 2: 20, 3: 30},
        )
        self.assertEqual({value[1] for value in results.values()}, {4})
        stats = executor.stats()
        self.assertEqual(stats["batches_total"], 1)
        self.assertEqual(stats["items_total"], 4)
        self.assertEqual(stats["max_observed_batch_size"], 4)

    def test_does_not_mix_incompatible_groups(self) -> None:
        observed: list[list[str]] = []

        def execute(values: list[str]) -> list[str]:
            observed.append(list(values))
            return values

        executor = MicrobatchExecutor(
            execute,
            max_batch_size=4,
            window_ms=20,
            max_queue_size=16,
        )
        results: list[str] = []

        def submit(group: str, value: str) -> None:
            results.append(executor.submit(group, value, timeout=2.0).value)

        threads = [
            threading.Thread(target=submit, args=("a", "a1")),
            threading.Thread(target=submit, args=("b", "b1")),
            threading.Thread(target=submit, args=("a", "a2")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2.0)
        executor.close()

        self.assertCountEqual(results, ["a1", "a2", "b1"])
        self.assertTrue(
            all({value[0] for value in batch} in ({"a"}, {"b"}) for batch in observed)
        )

    def test_fans_out_executor_errors(self) -> None:
        def execute(_values: list[int]) -> list[int]:
            raise ValueError("boom")

        executor = MicrobatchExecutor(
            execute,
            max_batch_size=2,
            window_ms=30,
            max_queue_size=4,
        )
        errors: list[str] = []

        def submit(value: int) -> None:
            try:
                executor.submit("same", value, timeout=2.0)
            except ValueError as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=submit, args=(value,)) for value in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2.0)
        executor.close()

        self.assertEqual(errors, ["boom", "boom"])

    def test_timeout_cancels_a_queued_item(self) -> None:
        gate = threading.Event()

        def execute(values: list[int]) -> list[int]:
            gate.wait(timeout=1.0)
            return values

        executor = MicrobatchExecutor(
            execute,
            max_batch_size=1,
            window_ms=0,
            max_queue_size=4,
        )
        first = threading.Thread(target=lambda: executor.submit("same", 1, timeout=1.0))
        first.start()
        time.sleep(0.02)
        with self.assertRaises(TimeoutError):
            executor.submit("same", 2, timeout=0.01)
        gate.set()
        first.join(timeout=1.0)
        executor.close()


if __name__ == "__main__":
    unittest.main()
