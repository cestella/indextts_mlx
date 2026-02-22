"""Tests for heartbeat expiry logic."""

import time

from indextts_mlx.srv.queue import JobQueue


def test_get_refreshes_heartbeat():
    q = JobQueue(heartbeat_timeout=300.0)
    job_id = q.submit("mock", "default", "app1", 10, {})
    job = q.get(job_id)
    t1 = job.last_heartbeat

    time.sleep(0.01)
    job = q.get(job_id)
    assert job.last_heartbeat > t1


def test_expired_job_skipped():
    q = JobQueue(heartbeat_timeout=0.01)
    job_id = q.submit("mock", "default", "app1", 10, {})
    time.sleep(0.02)

    # Should be expired, so pick_next returns None
    job = q.pick_next(None, None)
    assert job is None


def test_fresh_job_not_skipped():
    q = JobQueue(heartbeat_timeout=300.0)
    job_id = q.submit("mock", "default", "app1", 10, {})

    job = q.pick_next(None, None)
    assert job is not None
    assert job.id == job_id


def test_expired_status():
    q = JobQueue(heartbeat_timeout=0.01)
    job_id = q.submit("mock", "default", "app1", 10, {})
    time.sleep(0.02)

    # Trigger expiry check via pick_next
    q.pick_next(None, None)

    job = q.get(job_id)
    assert job.status == "expired"
