from __future__ import annotations

import argparse
import asyncio
import json
import platform
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from stats import summarize


@dataclass
class ScenarioResult:
    concurrency: int
    requests: int
    completed: int
    failed: int
    wall_seconds: float
    p50_ms: float
    p95_ms: float
    p99_ms: float

    @property
    def rps(self) -> float:
        return self.completed / self.wall_seconds if self.wall_seconds else 0.0


async def _simulate_request(
    base_ms: float, jitter_ms: float, failure_rate: float
) -> tuple[bool, float]:
    latency = max(0.0, (base_ms + random.uniform(-jitter_ms, jitter_ms)) / 1000.0)
    start = time.perf_counter()
    await asyncio.sleep(latency)
    failed = random.random() < failure_rate
    return (not failed), time.perf_counter() - start


async def run_scenario(
    *,
    concurrency: int,
    requests: int,
    base_ms: float,
    jitter_ms: float,
    failure_rate: float,
) -> ScenarioResult:
    latencies: list[float] = []
    completed = 0
    failed = 0
    sem = asyncio.Semaphore(concurrency)

    async def run_one() -> None:
        nonlocal completed, failed
        async with sem:
            ok, latency = await _simulate_request(base_ms, jitter_ms, failure_rate)
            if ok:
                completed += 1
                latencies.append(latency)
            else:
                failed += 1

    start = time.perf_counter()
    await asyncio.gather(*[run_one() for _ in range(requests)])
    wall = time.perf_counter() - start
    s = summarize(latencies)
    return ScenarioResult(
        concurrency=concurrency,
        requests=requests,
        completed=completed,
        failed=failed,
        wall_seconds=wall,
        p50_ms=s.p50_ms,
        p95_ms=s.p95_ms,
        p99_ms=s.p99_ms,
    )


def render_markdown(results: list[ScenarioResult]) -> str:
    lines = [
        "# RAG API load benchmark",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')}_",
        "",
    ]
    lines.append(
        "| Concurrency | Requests | Success | Failed | RPS | P50 (ms) | P95 (ms) | P99 (ms) |"
    )
    lines.append(
        "|------------:|---------:|--------:|-------:|----:|---------:|---------:|---------:|"
    )
    for r in results:
        lines.append(
            f"| {r.concurrency} | {r.requests} | {r.completed} | {r.failed} | {r.rps:.1f} | "
            f"{r.p50_ms:.1f} | {r.p95_ms:.1f} | {r.p99_ms:.1f} |"
        )
    return "\n".join(lines) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", default="5,20,50")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--base-ms", type=float, default=70.0)
    parser.add_argument("--jitter-ms", type=float, default=20.0)
    parser.add_argument("--failure-rate", type=float, default=0.01)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default=str(Path(__file__).parent / "results" / "report.md"))
    args = parser.parse_args()

    requests = 60 if args.quick else args.requests
    concurrencies = [int(v) for v in args.concurrency.split(",")]
    results = [
        await run_scenario(
            concurrency=c,
            requests=requests,
            base_ms=args.base_ms,
            jitter_ms=args.jitter_ms,
            failure_rate=args.failure_rate,
        )
        for c in concurrencies
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(results), encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "host": platform.platform(),
                "python": platform.python_version(),
                "scenarios": [r.__dict__ | {"rps": r.rps} for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(main())
