"""Tests for indextts_mlx.web.scheduler.

Covers:
  - YAML config loading (valid, missing fields, bad structure)
  - Job execution (_run_job): log file creation, naming, header/footer content,
    stdout capture, non-zero exit recording
  - Scheduler setup (start_scheduler): cron trigger registration, bad cron
    expression rejection, max_instances / coalesce settings
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# ── helpers ───────────────────────────────────────────────────────────────────


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "scheduler.yaml"
    p.write_text(content)
    return p


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def log_dir(tmp_path: Path) -> Path:
    d = tmp_path / "logs"
    d.mkdir()
    return d


@pytest.fixture()
def simple_job(log_dir: Path):
    from indextts_mlx.web.scheduler import ScheduledJobConfig

    return ScheduledJobConfig(
        name="test_job",
        schedule="0 6 * * *",
        command="echo",
        args=["hello"],
        log_dir=str(log_dir),
    )


# ── load_scheduler_config ─────────────────────────────────────────────────────


class TestLoadSchedulerConfig:
    def test_parses_minimal_valid_yaml(self, tmp_path: Path, log_dir: Path) -> None:
        yaml_path = _write_yaml(
            tmp_path,
            f"""
jobs:
  - name: gen_podcast
    schedule: "0 6 * * *"
    command: /usr/bin/make_podcast.sh
    log_dir: {log_dir}
""",
        )
        from indextts_mlx.web.scheduler import load_scheduler_config

        jobs = load_scheduler_config(yaml_path)
        assert len(jobs) == 1
        assert jobs[0].name == "gen_podcast"
        assert jobs[0].schedule == "0 6 * * *"
        assert jobs[0].command == "/usr/bin/make_podcast.sh"
        assert jobs[0].args == []
        assert jobs[0].log_dir == str(log_dir)

    def test_parses_args_list(self, tmp_path: Path, log_dir: Path) -> None:
        yaml_path = _write_yaml(
            tmp_path,
            f"""
jobs:
  - name: myjob
    schedule: "30 2 * * 0"
    command: /bin/script.sh
    args:
      - --output-dir
      - /some/path
      - --verbose
    log_dir: {log_dir}
""",
        )
        from indextts_mlx.web.scheduler import load_scheduler_config

        jobs = load_scheduler_config(yaml_path)
        assert jobs[0].args == ["--output-dir", "/some/path", "--verbose"]

    def test_parses_multiple_jobs(self, tmp_path: Path, log_dir: Path) -> None:
        yaml_path = _write_yaml(
            tmp_path,
            f"""
jobs:
  - name: job_a
    schedule: "0 6 * * *"
    command: /bin/a.sh
    log_dir: {log_dir}
  - name: job_b
    schedule: "0 22 * * *"
    command: /bin/b.sh
    log_dir: {log_dir}
""",
        )
        from indextts_mlx.web.scheduler import load_scheduler_config

        jobs = load_scheduler_config(yaml_path)
        assert len(jobs) == 2
        assert jobs[0].name == "job_a"
        assert jobs[1].name == "job_b"

    def test_empty_jobs_list_returns_empty(self, tmp_path: Path) -> None:
        yaml_path = _write_yaml(tmp_path, "jobs: []\n")
        from indextts_mlx.web.scheduler import load_scheduler_config

        assert load_scheduler_config(yaml_path) == []

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "scheduler.yaml"
        yaml_path.write_text("")
        from indextts_mlx.web.scheduler import load_scheduler_config

        assert load_scheduler_config(yaml_path) == []

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        from indextts_mlx.web.scheduler import load_scheduler_config

        with pytest.raises(FileNotFoundError):
            load_scheduler_config(tmp_path / "missing.yaml")

    @pytest.mark.parametrize("missing_field", ["name", "schedule", "command", "log_dir"])
    def test_raises_on_missing_required_field(
        self, tmp_path: Path, log_dir: Path, missing_field: str
    ) -> None:
        fields = {
            "name": "myjob",
            "schedule": '"0 6 * * *"',
            "command": "/bin/script.sh",
            "log_dir": str(log_dir),
        }
        fields.pop(missing_field)
        job_yaml = "\n".join(f"    {k}: {v}" for k, v in fields.items())
        yaml_path = _write_yaml(tmp_path, f"jobs:\n  -\n{job_yaml}\n")
        from indextts_mlx.web.scheduler import load_scheduler_config

        with pytest.raises(ValueError, match=missing_field):
            load_scheduler_config(yaml_path)

    def test_raises_if_jobs_not_a_list(self, tmp_path: Path) -> None:
        yaml_path = _write_yaml(tmp_path, "jobs: not_a_list\n")
        from indextts_mlx.web.scheduler import load_scheduler_config

        with pytest.raises(ValueError, match="'jobs' list"):
            load_scheduler_config(yaml_path)

    def test_args_coerced_to_strings(self, tmp_path: Path, log_dir: Path) -> None:
        yaml_path = _write_yaml(
            tmp_path,
            f"""
jobs:
  - name: myjob
    schedule: "0 0 * * *"
    command: /bin/script.sh
    args:
      - 42
      - 3.14
    log_dir: {log_dir}
""",
        )
        from indextts_mlx.web.scheduler import load_scheduler_config

        jobs = load_scheduler_config(yaml_path)
        assert jobs[0].args == ["42", "3.14"]


# ── _run_job ──────────────────────────────────────────────────────────────────


class TestRunJob:
    def test_creates_log_file_in_log_dir(self, simple_job, log_dir: Path) -> None:
        from indextts_mlx.web.scheduler import _run_job

        _run_job(simple_job)
        logs = list(log_dir.glob("test_job_*.log"))
        assert len(logs) == 1

    def test_log_filename_starts_with_job_name(self, simple_job, log_dir: Path) -> None:
        from indextts_mlx.web.scheduler import _run_job

        _run_job(simple_job)
        log_file = next(log_dir.glob("test_job_*.log"))
        assert log_file.name.startswith("test_job_")

    def test_log_filename_timestamp_format(self, simple_job, log_dir: Path) -> None:
        """Timestamp portion must match YYYY-MM-DDTHH-MM-SS."""
        import re

        from indextts_mlx.web.scheduler import _run_job

        _run_job(simple_job)
        log_file = next(log_dir.glob("test_job_*.log"))
        # strip "test_job_" prefix and ".log" suffix
        ts_part = log_file.stem[len("test_job_"):]
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}", ts_part), (
            f"Timestamp {ts_part!r} does not match expected format"
        )

    def test_log_contains_header_fields(self, simple_job, log_dir: Path) -> None:
        from indextts_mlx.web.scheduler import _run_job

        _run_job(simple_job)
        log_file = next(log_dir.glob("test_job_*.log"))
        text = log_file.read_text()
        assert "# job:" in text
        assert "test_job" in text
        assert "# started:" in text
        assert "# command:" in text
        assert "echo" in text

    def test_log_contains_footer_with_exit_code(self, simple_job, log_dir: Path) -> None:
        from indextts_mlx.web.scheduler import _run_job

        _run_job(simple_job)
        log_file = next(log_dir.glob("test_job_*.log"))
        text = log_file.read_text()
        assert "# finished:" in text
        assert "exit=0" in text

    def test_log_captures_stdout(self, log_dir: Path) -> None:
        from indextts_mlx.web.scheduler import ScheduledJobConfig, _run_job

        job = ScheduledJobConfig(
            name="echo_job",
            schedule="0 0 * * *",
            command=sys.executable,
            args=["-c", "print('hello from subprocess')"],
            log_dir=str(log_dir),
        )
        _run_job(job)
        log_file = next(log_dir.glob("echo_job_*.log"))
        assert "hello from subprocess" in log_file.read_text()

    def test_non_zero_exit_recorded_in_log(self, log_dir: Path) -> None:
        from indextts_mlx.web.scheduler import ScheduledJobConfig, _run_job

        job = ScheduledJobConfig(
            name="fail_job",
            schedule="0 0 * * *",
            command=sys.executable,
            args=["-c", "import sys; sys.exit(42)"],
            log_dir=str(log_dir),
        )
        _run_job(job)
        log_file = next(log_dir.glob("fail_job_*.log"))
        assert "exit=42" in log_file.read_text()

    def test_creates_log_dir_if_missing(self, tmp_path: Path) -> None:
        from indextts_mlx.web.scheduler import ScheduledJobConfig, _run_job

        missing_log_dir = tmp_path / "new" / "nested" / "logs"
        assert not missing_log_dir.exists()

        job = ScheduledJobConfig(
            name="mkdir_job",
            schedule="0 0 * * *",
            command=sys.executable,
            args=["-c", "pass"],
            log_dir=str(missing_log_dir),
        )
        _run_job(job)
        assert missing_log_dir.exists()
        assert list(missing_log_dir.glob("mkdir_job_*.log"))

    def test_launch_error_recorded_in_log(self, log_dir: Path) -> None:
        from indextts_mlx.web.scheduler import ScheduledJobConfig, _run_job

        job = ScheduledJobConfig(
            name="bad_cmd_job",
            schedule="0 0 * * *",
            command="/nonexistent/binary/that/does/not/exist",
            args=[],
            log_dir=str(log_dir),
        )
        # Should not raise — error goes into the log
        _run_job(job)
        log_file = next(log_dir.glob("bad_cmd_job_*.log"))
        text = log_file.read_text()
        assert "ERROR" in text

    def test_multiple_runs_produce_separate_logs(self, log_dir: Path) -> None:
        """Two runs of the same job create two distinct log files."""
        import time as _time

        from indextts_mlx.web.scheduler import ScheduledJobConfig, _run_job

        job = ScheduledJobConfig(
            name="multi_job",
            schedule="0 0 * * *",
            command=sys.executable,
            args=["-c", "pass"],
            log_dir=str(log_dir),
        )
        _run_job(job)
        _time.sleep(1.1)   # ensure timestamp differs by at least 1 second
        _run_job(job)
        logs = list(log_dir.glob("multi_job_*.log"))
        assert len(logs) == 2
        assert logs[0].name != logs[1].name

    def test_args_passed_to_subprocess(self, log_dir: Path) -> None:
        from indextts_mlx.web.scheduler import ScheduledJobConfig, _run_job

        sentinel = "SENTINEL_VALUE_12345"
        job = ScheduledJobConfig(
            name="args_job",
            schedule="0 0 * * *",
            command=sys.executable,
            args=["-c", f"print('{sentinel}')"],
            log_dir=str(log_dir),
        )
        _run_job(job)
        log_file = next(log_dir.glob("args_job_*.log"))
        assert sentinel in log_file.read_text()


# ── start_scheduler ──────────────────────────────────────────────────────────


apscheduler_available = False
try:
    import apscheduler  # noqa: F401
    apscheduler_available = True
except ImportError:
    pass

pytestmark_apscheduler = pytest.mark.skipif(
    not apscheduler_available, reason="apscheduler not installed"
)


class TestStartScheduler:
    @pytestmark_apscheduler
    def test_returns_running_scheduler(self, simple_job) -> None:
        from indextts_mlx.web.scheduler import start_scheduler

        sched = start_scheduler([simple_job])
        try:
            assert sched.running
        finally:
            sched.shutdown(wait=False)

    @pytestmark_apscheduler
    def test_registers_one_job_per_config(self, log_dir: Path) -> None:
        from indextts_mlx.web.scheduler import ScheduledJobConfig, start_scheduler

        jobs = [
            ScheduledJobConfig("a", "0 1 * * *", "echo", [], str(log_dir)),
            ScheduledJobConfig("b", "0 2 * * *", "echo", [], str(log_dir)),
        ]
        sched = start_scheduler(jobs)
        try:
            job_ids = {j.id for j in sched.get_jobs()}
            assert job_ids == {"a", "b"}
        finally:
            sched.shutdown(wait=False)

    @pytestmark_apscheduler
    def test_job_ids_match_config_names(self, simple_job) -> None:
        from indextts_mlx.web.scheduler import start_scheduler

        sched = start_scheduler([simple_job])
        try:
            ids = [j.id for j in sched.get_jobs()]
            assert "test_job" in ids
        finally:
            sched.shutdown(wait=False)

    @pytestmark_apscheduler
    def test_invalid_cron_raises_value_error(self, log_dir: Path) -> None:
        from indextts_mlx.web.scheduler import ScheduledJobConfig, start_scheduler

        bad_job = ScheduledJobConfig(
            name="bad",
            schedule="not a cron expression",
            command="echo",
            args=[],
            log_dir=str(log_dir),
        )
        with pytest.raises(ValueError, match="Invalid cron expression"):
            sched = start_scheduler([bad_job])
            sched.shutdown(wait=False)

    @pytestmark_apscheduler
    def test_max_instances_is_one(self, simple_job) -> None:
        """Each job should have max_instances=1 to prevent pileup."""
        from indextts_mlx.web.scheduler import start_scheduler

        sched = start_scheduler([simple_job])
        try:
            job = sched.get_job("test_job")
            assert job.max_instances == 1
        finally:
            sched.shutdown(wait=False)

    @pytestmark_apscheduler
    def test_empty_job_list_starts_cleanly(self) -> None:
        from indextts_mlx.web.scheduler import start_scheduler

        sched = start_scheduler([])
        try:
            assert sched.running
            assert sched.get_jobs() == []
        finally:
            sched.shutdown(wait=False)

    @pytestmark_apscheduler
    def test_shutdown_stops_scheduler(self, simple_job) -> None:
        from indextts_mlx.web.scheduler import start_scheduler

        sched = start_scheduler([simple_job])
        assert sched.running
        sched.shutdown(wait=False)
        assert not sched.running

    @pytestmark_apscheduler
    def test_worker_keyword_accepted(self, simple_job) -> None:
        """start_scheduler must accept worker= without TypeError."""
        from unittest.mock import MagicMock

        from indextts_mlx.web.scheduler import start_scheduler

        worker = MagicMock()
        sched = start_scheduler([simple_job], worker=worker)
        try:
            assert sched.running
        finally:
            sched.shutdown(wait=False)

    @pytestmark_apscheduler
    def test_worker_none_is_default(self, simple_job) -> None:
        """Omitting worker= (None) must not cause any error."""
        from indextts_mlx.web.scheduler import start_scheduler

        sched = start_scheduler([simple_job])
        try:
            assert sched.running
        finally:
            sched.shutdown(wait=False)


class TestRunJobWorkerIntegration:
    """Tests that _run_job correctly calls pause/resume on the worker."""

    def test_pause_called_before_subprocess(self, log_dir: Path) -> None:
        from unittest.mock import MagicMock, call

        from indextts_mlx.web.scheduler import ScheduledJobConfig, _run_job

        worker = MagicMock()
        calls = []
        worker.pause_for_scheduler.side_effect = lambda: calls.append("pause")
        worker.resume_for_scheduler.side_effect = lambda: calls.append("resume")

        job = ScheduledJobConfig(
            name="pause_job",
            schedule="0 0 * * *",
            command=sys.executable,
            args=["-c", "pass"],
            log_dir=str(log_dir),
        )
        _run_job(job, worker=worker)

        assert calls[0] == "pause", "pause_for_scheduler must be called before the subprocess"
        assert calls[-1] == "resume", "resume_for_scheduler must be called after the subprocess"

    def test_resume_called_even_on_launch_error(self, log_dir: Path) -> None:
        """resume_for_scheduler is called in the finally block even if the command fails."""
        from unittest.mock import MagicMock

        from indextts_mlx.web.scheduler import ScheduledJobConfig, _run_job

        worker = MagicMock()
        job = ScheduledJobConfig(
            name="error_job",
            schedule="0 0 * * *",
            command="/nonexistent/binary",
            args=[],
            log_dir=str(log_dir),
        )
        _run_job(job, worker=worker)

        worker.pause_for_scheduler.assert_called_once()
        worker.resume_for_scheduler.assert_called_once()

    def test_no_worker_runs_without_error(self, log_dir: Path) -> None:
        """_run_job(job, worker=None) runs the subprocess without calling any worker methods."""
        from indextts_mlx.web.scheduler import ScheduledJobConfig, _run_job

        job = ScheduledJobConfig(
            name="no_worker_job",
            schedule="0 0 * * *",
            command=sys.executable,
            args=["-c", "pass"],
            log_dir=str(log_dir),
        )
        # Must not raise even though no worker is provided
        _run_job(job, worker=None)
