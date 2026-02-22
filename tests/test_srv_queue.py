"""Tests for the priority job queue."""

import uuid

from indextts_mlx.srv.queue import JobQueue


def test_fifo_within_priority():
    q = JobQueue()
    id1 = q.submit("mock", "default", "app1", 10, {})
    id2 = q.submit("mock", "default", "app1", 10, {})
    id3 = q.submit("mock", "default", "app1", 10, {})

    job = q.pick_next(None, None)
    assert job.id == id1
    q.mark_done(id1, {})
    job = q.pick_next(None, None)
    assert job.id == id2
    q.mark_done(id2, {})
    job = q.pick_next(None, None)
    assert job.id == id3


def test_priority_ordering():
    q = JobQueue()
    id_low = q.submit("mock", "default", "app1", 10, {})
    id_urgent = q.submit("mock", "default", "app1", 0, {})

    job = q.pick_next(None, None)
    assert job.id == id_urgent


def test_app_grouping_sticky():
    q = JobQueue()
    id_a = q.submit("mock", "default", "appA", 10, {})
    id_b = q.submit("mock", "default", "appB", 10, {})
    id_a2 = q.submit("mock", "default", "appA", 10, {})

    # With current model loaded for appA, prefer appA jobs
    job = q.pick_next(("mock", "default"), "appA")
    assert job.id == id_a
    q.mark_done(id_a, {})

    job = q.pick_next(("mock", "default"), "appA")
    assert job.id == id_a2


def test_app_grouping_exhausted():
    q = JobQueue()
    id_a = q.submit("mock", "default", "appA", 10, {})
    id_b = q.submit("mock", "default", "appB", 10, {})

    q.mark_done(id_a, {})  # pretend appA done already — actually mark it
    # Actually let's do it properly:
    q2 = JobQueue()
    id_a = q2.submit("mock", "default", "appA", 10, {})
    id_b = q2.submit("mock", "default", "appB", 10, {})

    job = q2.pick_next(("mock", "default"), "appA")
    assert job.id == id_a
    q2.mark_done(id_a, {})

    # No more appA jobs — should pick appB
    job = q2.pick_next(("mock", "default"), "appA")
    assert job.id == id_b


def test_priority_0_ignores_grouping():
    q = JobQueue()
    id_a = q.submit("mock", "default", "appA", 10, {})
    id_urgent = q.submit("mock", "default", "appB", 0, {})

    # Even though current app is appA, urgent job wins
    job = q.pick_next(("mock", "default"), "appA")
    assert job.id == id_urgent


def test_cancel_queued():
    q = JobQueue()
    id1 = q.submit("mock", "default", "app1", 10, {})
    id2 = q.submit("mock", "default", "app1", 10, {})

    assert q.cancel(id1)

    job = q.pick_next(None, None)
    assert job.id == id2


def test_submit_returns_id():
    q = JobQueue()
    job_id = q.submit("mock", "default", "app1", 10, {})
    # Should be a valid UUID
    uuid.UUID(job_id)


def test_max_queue_size():
    q = JobQueue(max_size=2)
    q.submit("mock", "default", "app1", 10, {})
    q.submit("mock", "default", "app1", 10, {})

    try:
        q.submit("mock", "default", "app1", 10, {})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "full" in str(e).lower()
