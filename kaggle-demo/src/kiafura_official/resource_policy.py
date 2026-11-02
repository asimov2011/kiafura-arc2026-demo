"""Host resource limits for long-running Kiafura experiment workers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CpuPolicy:
    """Conservative proxy for half of the host's six masked P-cores."""

    worker_cpus: tuple[int, ...] = (0, 2, 4)
    max_per_core: float = 0.67
    period_seconds: float = 0.15

    def validate(self) -> None:
        allowed = os.sched_getaffinity(0)
        if not self.worker_cpus or not set(self.worker_cpus) <= allowed:
            raise RuntimeError(
                f"worker CPUs {self.worker_cpus!r} are not available in {sorted(allowed)!r}"
            )
        if len({cpu // 2 for cpu in self.worker_cpus}) != len(self.worker_cpus):
            raise RuntimeError("worker CPUs must map to distinct exposed physical cores")
        if not 0.0 < self.max_per_core <= 0.67:
            raise RuntimeError("per-core duty cycle must be in (0, 0.67]")
        if self.period_seconds <= 0.0:
            raise RuntimeError("period must be positive")


POLICY = CpuPolicy()


@dataclass(frozen=True)
class GovernedResult:
    returncode: int
    stdout: str
    stderr: str
    active_seconds: float
    wall_seconds: float


def run_capped(
    command: Sequence[str],
    *,
    cpu: int,
    cwd: Path,
    env: Mapping[str, str],
    policy: CpuPolicy = POLICY,
) -> GovernedResult:
    """Run one process pinned to one core with a bounded duty cycle.

    The worker is a leaf process, so limiting it also limits its workload.
    There is deliberately no timeout: completion or a real process failure are
    the only terminal conditions.
    """

    policy.validate()
    if cpu not in policy.worker_cpus:
        raise ValueError(f"CPU {cpu} is outside the configured worker set")
    started = time.monotonic()
    active = 0.0
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    os.sched_setaffinity(process.pid, {cpu})
    running = True
    try:
        while process.poll() is None:
            if not running:
                os.kill(process.pid, signal.SIGCONT)
                running = True
            active_slice = policy.period_seconds * policy.max_per_core
            slice_started = time.monotonic()
            time.sleep(active_slice)
            active += time.monotonic() - slice_started
            if process.poll() is not None:
                break
            os.kill(process.pid, signal.SIGSTOP)
            running = False
            time.sleep(policy.period_seconds - active_slice)
        stdout, stderr = process.communicate()
    finally:
        if process.poll() is None:
            if not running:
                os.kill(process.pid, signal.SIGCONT)
            process.kill()
            process.wait()
    return GovernedResult(
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
        active_seconds=active,
        wall_seconds=time.monotonic() - started,
    )
