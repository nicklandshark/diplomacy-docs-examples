from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from tinker_cookbook.rl import train as rl_train


class _FakeLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, float], int]] = []

    def log_metrics(self, metrics, step: int) -> None:
        self.calls.append((dict(metrics), step))


class _FakeBuilder:
    def __init__(self, *, requires_in_process_rollout: bool) -> None:
        self.requires_in_process_rollout = requires_in_process_rollout


class _FakeDataset:
    def __init__(self, builders_by_batch: list[list[object]]) -> None:
        self._builders_by_batch = builders_by_batch

    def get_batch(self, index: int):
        return self._builders_by_batch[index]

    def __len__(self) -> int:
        return len(self._builders_by_batch)


def test_log_metrics_merges_extra_rollout_metrics() -> None:
    cfg = rl_train.Config(
        learning_rate=1e-5,
        dataset_builder=object(),
        model_name="Qwen/Qwen3.5-27B",
        max_tokens=32,
        log_path="/tmp/diplomacy-test",
        extra_metrics_provider=lambda: {
            "rollout/runner_call_count": 3.0,
            "rollout/runner_failure_count": 1.0,
        },
    )
    fake_logger = _FakeLogger()

    rl_train._log_metrics(fake_logger, cfg, {"train/loss": 0.25}, step=7)

    assert fake_logger.calls == [
        (
            {
                "train/loss": 0.25,
                "rollout/runner_call_count": 3.0,
                "rollout/runner_failure_count": 1.0,
            },
            7,
        )
    ]


def test_rollout_executor_rejected_for_in_process_only_builders() -> None:
    dataset = _FakeDataset([[ _FakeBuilder(requires_in_process_rollout=True) ]])

    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            rl_train._ensure_rollout_executor_compatible(executor, dataset, None)
        except ValueError as exc:
            assert "require in-process execution" in str(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("Expected rollout executor compatibility check to fail")


def test_rollout_executor_allowed_for_picklable_builders() -> None:
    dataset = _FakeDataset([[ _FakeBuilder(requires_in_process_rollout=False) ]])

    with ThreadPoolExecutor(max_workers=1) as executor:
        rl_train._ensure_rollout_executor_compatible(executor, dataset, None)
