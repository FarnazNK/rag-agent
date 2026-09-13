"""Latency statistics.

Separated from the runner so the percentile logic is unit-testable. Tail
latency is the entire point of this benchmark, and a p99 computed wrong is
worse than no p99 — it looks authoritative and is quietly false.

Two things this module is careful about:

1. **Interpolation.** numpy's default `linear` method interpolates between
   samples, inventing a value that never occurred. For latency reporting we
   use `lower` (nearest-rank), so every number reported is a measurement that
   actually happened.

2. **Sample count.** p99 over 50 samples is noise: it is determined by a
   single observation. `is_reliable` flags that rather than letting the
   number be quoted unqualified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class LatencySummary:
    """Percentile summary for one measurement series."""

    count: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    #: p99 needs ~100+ samples to mean anything. Below that it IS the max.
    p99_reliable: bool

    def as_dict(self) -> dict:
        return asdict(self)


def summarize(latencies_seconds: list[float]) -> LatencySummary:
    """Summarize latencies (given in seconds) into a millisecond report.

    Uses nearest-rank percentiles so every reported figure is a real
    observation rather than an interpolation between two.
    """
    if not latencies_seconds:
        return LatencySummary(
            count=0,
            mean_ms=0.0,
            p50_ms=0.0,
            p90_ms=0.0,
            p95_ms=0.0,
            p99_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
            p99_reliable=False,
        )

    arr = sorted(v * 1000.0 for v in latencies_seconds)

    def pct(q: float) -> float:
        idx = int((q / 100) * (len(arr) - 1))
        return float(arr[idx])

    return LatencySummary(
        count=len(arr),
        mean_ms=float(sum(arr) / len(arr)),
        p50_ms=pct(50),
        p90_ms=pct(90),
        p95_ms=pct(95),
        p99_ms=pct(99),
        min_ms=float(arr[0]),
        max_ms=float(arr[-1]),
        # 100 samples => p99 is the single worst observation. Call it
        # unreliable below that rather than printing it as though it means
        # something.
        p99_reliable=len(arr) >= 100,
    )


def work_factor(compute_seconds: float, work_units: float) -> float:
    """Compute seconds spent per unit of work."""
    if work_units <= 0:
        return 0.0
    return compute_seconds / work_units
