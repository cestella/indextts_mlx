"""Cron-scheduled command runner for the IndexTTS web backend.

Reads a YAML file that lists jobs with cron schedules, commands, arguments,
and a log directory.  Uses APScheduler (BackgroundScheduler + CronTrigger) for
scheduling and runs each job in a subprocess, writing timestamped log files.

YAML format::

    jobs:
      - name: generate_daily_podcast
        schedule: "0 6 * * *"   # standard 5-field cron: min hour dom month dow
        command: /path/to/script.sh
        args:
          - --output-dir
          - /podcasts/daily
        log_dir: /var/log/indextts/scheduler   # timestamped log per run

All fields except ``args`` are required.  ``args`` defaults to an empty list.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class ScheduledJobConfig:
    name: str
    schedule: str          # 5-field cron expression, e.g. "0 6 * * *"
    command: str
    args: List[str] = field(default_factory=list)
    log_dir: str = ""


def load_scheduler_config(yaml_path: str | Path) -> List[ScheduledJobConfig]:
    """Parse ``yaml_path`` and return a list of :class:`ScheduledJobConfig`.

    Raises ``ImportError`` if PyYAML is missing and ``ValueError`` on bad
    config structure.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for the scheduler. Install with: pip install pyyaml"
        ) from exc

    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Scheduler config not found: {path}")

    with path.open() as fh:
        data = yaml.safe_load(fh) or {}

    raw_jobs = data.get("jobs", [])
    if not isinstance(raw_jobs, list):
        raise ValueError("scheduler YAML must have a top-level 'jobs' list")

    jobs: List[ScheduledJobConfig] = []
    for i, entry in enumerate(raw_jobs):
        for key in ("name", "schedule", "command", "log_dir"):
            if not entry.get(key):
                raise ValueError(
                    f"scheduler job [{i}] is missing required field '{key}'"
                )
        jobs.append(
            ScheduledJobConfig(
                name=entry["name"],
                schedule=entry["schedule"],
                command=entry["command"],
                args=[str(a) for a in (entry.get("args") or [])],
                log_dir=entry["log_dir"],
            )
        )
    return jobs


# ── Job execution ─────────────────────────────────────────────────────────────


def _run_job(job: ScheduledJobConfig, worker=None) -> None:
    """Execute *job* and write all output to a timestamped log file.

    If *worker* is provided, the currently running TTS job (if any) is
    interrupted and requeued before the script starts, and the worker is
    unblocked again after the script finishes.
    """
    if worker is not None:
        worker.pause_for_scheduler()

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_dir = Path(job.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job.name}_{ts}.log"

    cmd = [job.command] + job.args

    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as fh:
            fh.write(f"# job:     {job.name}\n")
            fh.write(f"# started: {datetime.now().isoformat()}\n")
            fh.write(f"# command: {' '.join(cmd)}\n\n")
            fh.flush()

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                )
                returncode = proc.wait()
            except Exception as exc:
                fh.write(f"\n# ERROR launching process: {exc}\n")
                returncode = -1

            fh.write(f"\n# finished: {datetime.now().isoformat()}")
            fh.write(f"  exit={returncode}\n")
    finally:
        if worker is not None:
            worker.resume_for_scheduler()

    # Mirror a brief summary to stderr so it appears in the server log
    status = "OK" if returncode == 0 else f"FAILED (exit {returncode})"
    print(
        f"[scheduler] {job.name}: {status}  log={log_path}",
        file=sys.stderr,
        flush=True,
    )


# ── Scheduler lifecycle ───────────────────────────────────────────────────────


def start_scheduler(jobs: List[ScheduledJobConfig], worker=None):
    """Create, populate, and start an APScheduler BackgroundScheduler.

    If *worker* is provided, each scheduled job will pause the TTS worker
    (interrupting and requeueing any active job) before running the script,
    then resume it afterwards.

    Returns the running scheduler instance so the caller can call
    ``scheduler.shutdown()`` on exit.

    Raises ``ImportError`` if APScheduler is not installed.
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler  # type: ignore[import-untyped]
        from apscheduler.triggers.cron import CronTrigger  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "APScheduler is required for the scheduler. "
            "Install with: pip install 'apscheduler<4'"
        ) from exc

    scheduler = BackgroundScheduler(daemon=True)

    for job in jobs:
        try:
            trigger = CronTrigger.from_crontab(job.schedule)
        except Exception as exc:
            raise ValueError(
                f"Invalid cron expression for job '{job.name}': {job.schedule!r} — {exc}"
            ) from exc

        # Pass worker alongside job so _run_job can pause/resume the TTS worker.
        # Capture both by value in args to avoid late-binding closure bugs.
        scheduler.add_job(
            _run_job,
            trigger=trigger,
            args=[job, worker],
            id=job.name,
            name=job.name,
            replace_existing=True,
            max_instances=1,   # don't pile up if a run takes longer than the interval
            coalesce=True,     # skip missed fire if we were down
        )
        print(f"[scheduler] registered '{job.name}'  schedule={job.schedule!r}", file=sys.stderr)

    scheduler.start()
    return scheduler
