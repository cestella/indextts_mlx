"""Tests for SrvClient — unit tests with mocked httpx + integration tests."""

from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
import uvicorn

from indextts_mlx.srv.client import SrvClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(json_data: dict, status_code: int = 200) -> httpx.Response:
    """Build a fake httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    return resp


@pytest.fixture()
def client():
    """SrvClient with a mocked internal httpx.Client."""
    with patch("indextts_mlx.srv.client.httpx.Client") as MockClient:
        mock_http = MagicMock()
        MockClient.return_value = mock_http
        c = SrvClient(socket_path="/tmp/test.sock")
        c._mock_http = mock_http  # expose for assertions
        yield c


# ---------------------------------------------------------------------------
# Low-level
# ---------------------------------------------------------------------------

class TestSubmit:
    def test_returns_job_id(self, client):
        client._mock_http.post.return_value = _mock_response({"job_id": "abc-123"})
        job_id = client.submit(model_type="llm", payload={"prompt": "hi"})
        assert job_id == "abc-123"
        call_args = client._mock_http.post.call_args
        assert call_args[0][0] == "/jobs"
        body = call_args[1]["json"]
        assert body["model_type"] == "llm"
        assert body["payload"] == {"prompt": "hi"}

    def test_passes_optional_fields(self, client):
        client._mock_http.post.return_value = _mock_response({"job_id": "x"})
        client.submit(
            model_type="tts_indextts",
            payload={},
            model="large",
            application_id="app1",
            priority=0,
            result_path="/tmp/out.wav",
        )
        body = client._mock_http.post.call_args[1]["json"]
        assert body["model"] == "large"
        assert body["application_id"] == "app1"
        assert body["priority"] == 0
        assert body["result_path"] == "/tmp/out.wav"


class TestGetJob:
    def test_returns_job_dict(self, client):
        job_data = {"id": "j1", "status": "queued"}
        client._mock_http.get.return_value = _mock_response(job_data)
        result = client.get_job("j1")
        assert result == job_data
        client._mock_http.get.assert_called_once_with("/jobs/j1")


class TestWait:
    def test_polls_until_done(self, client):
        responses = [
            _mock_response({"id": "j1", "status": "queued"}),
            _mock_response({"id": "j1", "status": "running"}),
            _mock_response({"id": "j1", "status": "done", "result": {"text": "ok"}}),
        ]
        client._mock_http.get.side_effect = responses
        result = client.wait("j1", timeout=10, poll_interval=0.01)
        assert result["status"] == "done"
        assert result["result"] == {"text": "ok"}
        assert client._mock_http.get.call_count == 3

    def test_raises_on_failure(self, client):
        client._mock_http.get.return_value = _mock_response(
            {"id": "j1", "status": "failed", "error": "OOM"}
        )
        with pytest.raises(RuntimeError, match="failed.*OOM"):
            client.wait("j1", timeout=10, poll_interval=0.01)

    def test_raises_on_cancelled(self, client):
        client._mock_http.get.return_value = _mock_response(
            {"id": "j1", "status": "cancelled", "error": None}
        )
        with pytest.raises(RuntimeError, match="cancelled"):
            client.wait("j1", timeout=10, poll_interval=0.01)

    def test_raises_on_timeout(self, client):
        client._mock_http.get.return_value = _mock_response(
            {"id": "j1", "status": "queued"}
        )
        with pytest.raises(TimeoutError, match="did not complete"):
            client.wait("j1", timeout=0.05, poll_interval=0.01)


class TestCancel:
    def test_cancel_calls_delete(self, client):
        client._mock_http.delete.return_value = _mock_response({"status": "cancelled"})
        client.cancel("j1")
        client._mock_http.delete.assert_called_once_with("/jobs/j1")


class TestHealth:
    def test_health(self, client):
        data = {"status": "ok", "queue_depth": 2}
        client._mock_http.get.return_value = _mock_response(data)
        assert client.health() == data


class TestQueue:
    def test_queue(self, client):
        data = {"active": None, "queued": [], "recent": []}
        client._mock_http.get.return_value = _mock_response(data)
        assert client.queue() == data


# ---------------------------------------------------------------------------
# Convenience methods
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_submits_and_waits(self, client):
        client._mock_http.post.return_value = _mock_response({"job_id": "g1"})
        client._mock_http.get.return_value = _mock_response(
            {"id": "g1", "status": "done", "result": {"text": "4"}}
        )
        result = client.generate(prompt="2+2", system_prompt="Be concise.", max_tokens=64)
        assert result["result"]["text"] == "4"
        # Verify payload
        body = client._mock_http.post.call_args[1]["json"]
        assert body["model_type"] == "llm"
        assert body["payload"]["prompt"] == "2+2"
        assert body["payload"]["system_prompt"] == "Be concise."
        assert body["payload"]["max_tokens"] == 64

    def test_extra_kwargs_passed(self, client):
        client._mock_http.post.return_value = _mock_response({"job_id": "g2"})
        client._mock_http.get.return_value = _mock_response(
            {"id": "g2", "status": "done", "result": {}}
        )
        client.generate(prompt="hi", temperature=0.5)
        body = client._mock_http.post.call_args[1]["json"]
        assert body["payload"]["temperature"] == 0.5


class TestSynthesize:
    def test_submits_and_waits(self, client):
        client._mock_http.post.return_value = _mock_response({"job_id": "s1"})
        client._mock_http.get.return_value = _mock_response(
            {"id": "s1", "status": "done", "result": {"result_path": "/tmp/out.wav"}}
        )
        result = client.synthesize(
            text="Hello.",
            result_path="/tmp/out.wav",
            spk_audio_prompt="speaker.wav",
            seed=42,
        )
        assert result["result"]["result_path"] == "/tmp/out.wav"
        body = client._mock_http.post.call_args[1]["json"]
        assert body["model_type"] == "tts_indextts"
        assert body["payload"]["text"] == "Hello."
        assert body["payload"]["spk_audio_prompt"] == "speaker.wav"
        assert body["payload"]["seed"] == 42
        assert body["result_path"] == "/tmp/out.wav"

    def test_optional_tts_params(self, client):
        client._mock_http.post.return_value = _mock_response({"job_id": "s2"})
        client._mock_http.get.return_value = _mock_response(
            {"id": "s2", "status": "done", "result": {}}
        )
        client.synthesize(
            text="Hi",
            result_path="/tmp/x.wav",
            voice="emma",
            voices_dir="/voices",
            emotion=1.5,
            cfm_steps=25,
            emo_vector="0.8,0,0,0,0,0,0.2,0",
        )
        body = client._mock_http.post.call_args[1]["json"]
        assert body["payload"]["voice"] == "emma"
        assert body["payload"]["voices_dir"] == "/voices"
        assert body["payload"]["emotion"] == 1.5
        assert body["payload"]["cfm_steps"] == 25
        assert body["payload"]["emo_vector"] == "0.8,0,0,0,0,0,0.2,0"


class TestTranscribe:
    def test_submits_and_waits(self, client):
        client._mock_http.post.return_value = _mock_response({"job_id": "t1"})
        client._mock_http.get.return_value = _mock_response(
            {"id": "t1", "status": "done", "result": {"text": "hello world"}}
        )
        result = client.transcribe(audio_path="/tmp/audio.mp3", language="en")
        assert result["result"]["text"] == "hello world"
        body = client._mock_http.post.call_args[1]["json"]
        assert body["model_type"] == "whisperx"
        assert body["payload"]["audio_path"] == "/tmp/audio.mp3"
        assert body["payload"]["language"] == "en"


class TestTranslate:
    def test_submits_and_waits(self, client):
        client._mock_http.post.return_value = _mock_response({"job_id": "tr1"})
        client._mock_http.get.return_value = _mock_response(
            {"id": "tr1", "status": "done", "result": {"text": "Good morning"}}
        )
        result = client.translate(text="Buongiorno", src_lang="ita", tgt_lang="eng")
        assert result["result"]["text"] == "Good morning"
        body = client._mock_http.post.call_args[1]["json"]
        assert body["model_type"] == "translation"
        assert body["payload"]["text"] == "Buongiorno"
        assert body["payload"]["src_lang"] == "ita"
        assert body["payload"]["tgt_lang"] == "eng"


# ---------------------------------------------------------------------------
# Voice cloning / custom voice / voice design (engine-based)
# ---------------------------------------------------------------------------


class TestCloneVoice:
    def test_qwen3_submits_and_waits(self, client):
        client._mock_http.post.return_value = _mock_response({"job_id": "cv1"})
        client._mock_http.get.return_value = _mock_response(
            {"id": "cv1", "status": "done", "result": {"result_path": "/tmp/out.wav"}}
        )
        result = client.clone_voice(
            text="Hello.",
            ref_audio="speaker.wav",
            result_path="/tmp/out.wav",
            engine="qwen3-tts",
            ref_text="reference text",
            language="english",
        )
        assert result["status"] == "done"
        body = client._mock_http.post.call_args[1]["json"]
        assert body["model_type"] == "tts_qwen3"
        assert body["model"] == "base"
        assert body["payload"]["mode"] == "clone"
        assert body["payload"]["ref_audio"] == "speaker.wav"
        assert body["payload"]["ref_text"] == "reference text"
        assert body["payload"]["language"] == "english"
        assert body["result_path"] == "/tmp/out.wav"

    def test_indextts_submits_and_waits(self, client):
        client._mock_http.post.return_value = _mock_response({"job_id": "cv2"})
        client._mock_http.get.return_value = _mock_response(
            {"id": "cv2", "status": "done", "result": {"result_path": "/tmp/out.wav"}}
        )
        result = client.clone_voice(
            text="Hello.",
            ref_audio="speaker.wav",
            result_path="/tmp/out.wav",
            engine="indextts",
            seed=42,
            use_random=False,
        )
        assert result["status"] == "done"
        body = client._mock_http.post.call_args[1]["json"]
        assert body["model_type"] == "tts_indextts"
        assert body["model"] == "indextts2"
        # indextts maps ref_audio → spk_audio_prompt
        assert body["payload"]["spk_audio_prompt"] == "speaker.wav"
        assert body["payload"]["seed"] == 42
        assert body["payload"]["use_random"] is False
        assert "ref_audio" not in body["payload"]

    def test_unknown_engine_raises(self, client):
        with pytest.raises(ValueError, match="Unknown clone engine"):
            client.submit_clone_voice(
                text="hi", ref_audio="s.wav", result_path="/tmp/x.wav", engine="nope"
            )

    def test_extra_kwargs_forwarded(self, client):
        client._mock_http.post.return_value = _mock_response({"job_id": "cv3"})
        client._mock_http.get.return_value = _mock_response(
            {"id": "cv3", "status": "done", "result": {}}
        )
        client.clone_voice(
            text="Hi",
            ref_audio="s.wav",
            result_path="/tmp/x.wav",
            engine="qwen3-tts",
            temperature=0.7,
            top_k=50,
        )
        body = client._mock_http.post.call_args[1]["json"]
        assert body["payload"]["temperature"] == 0.7
        assert body["payload"]["top_k"] == 50


class TestCustomVoice:
    def test_submits_and_waits(self, client):
        client._mock_http.post.return_value = _mock_response({"job_id": "cust1"})
        client._mock_http.get.return_value = _mock_response(
            {"id": "cust1", "status": "done", "result": {"result_path": "/tmp/out.wav"}}
        )
        result = client.custom_voice(
            text="I'm excited!",
            speaker="Vivian",
            result_path="/tmp/out.wav",
            instruct="Very happy and excited.",
            language="english",
        )
        assert result["status"] == "done"
        body = client._mock_http.post.call_args[1]["json"]
        assert body["model_type"] == "tts_qwen3"
        assert body["model"] == "custom-voice"
        assert body["payload"]["mode"] == "custom_voice"
        assert body["payload"]["speaker"] == "Vivian"
        assert body["payload"]["instruct"] == "Very happy and excited."

    def test_unknown_engine_raises(self, client):
        with pytest.raises(ValueError, match="Unknown custom_voice engine"):
            client.submit_custom_voice(
                text="hi", speaker="X", result_path="/tmp/x.wav", engine="nope"
            )


class TestDesignVoice:
    def test_submits_and_waits(self, client):
        client._mock_http.post.return_value = _mock_response({"job_id": "dv1"})
        client._mock_http.get.return_value = _mock_response(
            {"id": "dv1", "status": "done", "result": {"result_path": "/tmp/out.wav"}}
        )
        result = client.design_voice(
            text="La lluvia caia suavemente.",
            instruct="A calm Univision news broadcaster",
            result_path="/tmp/out.wav",
            language="spanish",
        )
        assert result["status"] == "done"
        body = client._mock_http.post.call_args[1]["json"]
        assert body["model_type"] == "tts_qwen3"
        assert body["model"] == "voice-design"
        assert body["payload"]["mode"] == "voice_design"
        assert body["payload"]["instruct"] == "A calm Univision news broadcaster"
        assert body["payload"]["language"] == "spanish"

    def test_unknown_engine_raises(self, client):
        with pytest.raises(ValueError, match="Unknown design_voice engine"):
            client.submit_design_voice(
                text="hi", instruct="X", result_path="/tmp/x.wav", engine="nope"
            )


# ---------------------------------------------------------------------------
# CPU endpoints
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_direct_call(self, client):
        client._mock_http.post.return_value = _mock_response(
            {"text": "forty five dollars and fifty cents"}
        )
        result = client.normalize(text="$45.50", language="en")
        assert result["text"] == "forty five dollars and fifty cents"
        call_args = client._mock_http.post.call_args
        assert call_args[0][0] == "/cpu/normalize"
        assert call_args[1]["json"] == {"text": "$45.50", "language": "en"}


class TestSegment:
    def test_direct_call(self, client):
        client._mock_http.post.return_value = _mock_response(
            {"segments": ["chunk1", "chunk2"]}
        )
        result = client.segment(text="Long text", max_chars=100)
        assert result["segments"] == ["chunk1", "chunk2"]
        call_args = client._mock_http.post.call_args
        assert call_args[0][0] == "/cpu/segment"
        body = call_args[1]["json"]
        assert body["text"] == "Long text"
        assert body["max_chars"] == 100
        assert body["strategy"] == "char_count"
        assert body["language"] == "english"


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_close_called(self, client):
        with client:
            pass
        client._mock_http.close.assert_called_once()

    def test_usable_as_context_manager(self, client):
        with client as c:
            assert c is client


# ---------------------------------------------------------------------------
# Submit-only methods
# ---------------------------------------------------------------------------

class TestSubmitOnly:
    def test_submit_generate(self, client):
        client._mock_http.post.return_value = _mock_response({"job_id": "sg1"})
        job_id = client.submit_generate(prompt="hi")
        assert job_id == "sg1"

    def test_submit_synthesize(self, client):
        client._mock_http.post.return_value = _mock_response({"job_id": "ss1"})
        job_id = client.submit_synthesize(text="hi", result_path="/tmp/out.wav")
        assert job_id == "ss1"

    def test_submit_transcribe(self, client):
        client._mock_http.post.return_value = _mock_response({"job_id": "st1"})
        job_id = client.submit_transcribe(audio_path="/tmp/a.mp3")
        assert job_id == "st1"

    def test_submit_translate(self, client):
        client._mock_http.post.return_value = _mock_response({"job_id": "str1"})
        job_id = client.submit_translate(text="ciao")
        assert job_id == "str1"

    def test_submit_clone_voice(self, client):
        client._mock_http.post.return_value = _mock_response({"job_id": "scv1"})
        job_id = client.submit_clone_voice(
            text="hi", ref_audio="s.wav", result_path="/tmp/x.wav"
        )
        assert job_id == "scv1"

    def test_submit_custom_voice(self, client):
        client._mock_http.post.return_value = _mock_response({"job_id": "scust1"})
        job_id = client.submit_custom_voice(
            text="hi", speaker="Vivian", result_path="/tmp/x.wav"
        )
        assert job_id == "scust1"

    def test_submit_design_voice(self, client):
        client._mock_http.post.return_value = _mock_response({"job_id": "sdv1"})
        job_id = client.submit_design_voice(
            text="hi", instruct="A narrator", result_path="/tmp/x.wav"
        )
        assert job_id == "sdv1"


# ===========================================================================
# Integration tests — real FastAPI server with mock backend over Unix socket
# ===========================================================================


def _start_server(socket_path, models_config, log_level="error"):
    """Start a uvicorn server on a Unix socket, return (server, thread)."""
    from indextts_mlx.srv.app import create_app
    from indextts_mlx.srv.config import SrvConfig

    config = SrvConfig(socket_path=socket_path, heartbeat_timeout_s=300.0)
    app = create_app(config=config, models_config=models_config)

    server = uvicorn.Server(
        uvicorn.Config(app, uds=socket_path, log_level=log_level)
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for socket to appear
    deadline = time.monotonic() + 5.0
    while not os.path.exists(socket_path):
        if time.monotonic() >= deadline:
            raise RuntimeError("srv socket never appeared")
        time.sleep(0.05)
    # Give uvicorn a moment to start accepting
    time.sleep(0.1)
    return server, thread


@pytest.fixture()
def srv_socket():
    """Start a real srv FastAPI app on a temp Unix socket, yield socket path."""
    # Use /tmp directly to avoid AF_UNIX path length limit (104 bytes on macOS)
    sock_fd, socket_path = tempfile.mkstemp(prefix="srv_test_", suffix=".sock", dir="/tmp")
    os.close(sock_fd)
    os.unlink(socket_path)

    models_config = {
        "backends": {
            "mock": {
                "default": "default",
                "models": {"default": {}},
            },
        },
    }
    server, thread = _start_server(socket_path, models_config)

    yield socket_path

    server.should_exit = True
    thread.join(timeout=5.0)
    if os.path.exists(socket_path):
        os.unlink(socket_path)


class TestIntegrationMockBackend:
    """Integration tests using a real server with the mock backend."""

    def test_health(self, srv_socket):
        with SrvClient(socket_path=srv_socket) as c:
            h = c.health()
            assert h["status"] == "ok"
            assert "queue_depth" in h
            assert "uptime_s" in h

    def test_queue_empty(self, srv_socket):
        with SrvClient(socket_path=srv_socket) as c:
            q = c.queue()
            assert q["active"] is None
            assert q["queued"] == []

    def test_submit_and_wait(self, srv_socket):
        with SrvClient(socket_path=srv_socket) as c:
            job_id = c.submit(
                model_type="mock",
                payload={"duration": 0.01},
            )
            assert isinstance(job_id, str)
            assert len(job_id) > 0

            job = c.wait(job_id, timeout=10, poll_interval=0.05)
            assert job["status"] == "done"
            assert job["result"]["status"] == "ok"

    def test_submit_and_get_job(self, srv_socket):
        with SrvClient(socket_path=srv_socket) as c:
            job_id = c.submit(model_type="mock", payload={"duration": 0.01})
            job = c.get_job(job_id)
            assert job["id"] == job_id
            assert job["model_type"] == "mock"

    def test_cancel_queued_job(self, srv_socket):
        with SrvClient(socket_path=srv_socket) as c:
            # Submit a slow job first so the next one stays queued
            c.submit(model_type="mock", payload={"duration": 5.0})
            # This one should queue behind the slow job
            job_id = c.submit(model_type="mock", payload={"duration": 0.01})
            c.cancel(job_id)
            job = c.get_job(job_id)
            assert job["status"] == "cancelled"

    def test_result_path_written(self, srv_socket, tmp_path):
        result_file = tmp_path / "result.txt"
        with SrvClient(socket_path=srv_socket) as c:
            job_id = c.submit(
                model_type="mock",
                payload={"duration": 0.01, "result_path": str(result_file)},
            )
            c.wait(job_id, timeout=10, poll_interval=0.05)
            assert result_file.exists()
            assert result_file.read_text() == "mock result"

    def test_wait_timeout(self, srv_socket):
        with SrvClient(socket_path=srv_socket) as c:
            # Submit a very slow job
            job_id = c.submit(model_type="mock", payload={"duration": 60.0})
            with pytest.raises(TimeoutError):
                c.wait(job_id, timeout=0.2, poll_interval=0.05)

    def test_multiple_jobs_sequential(self, srv_socket):
        with SrvClient(socket_path=srv_socket) as c:
            results = []
            for i in range(3):
                job_id = c.submit(model_type="mock", payload={"duration": 0.01})
                job = c.wait(job_id, timeout=10, poll_interval=0.05)
                results.append(job)
            assert all(r["status"] == "done" for r in results)

    def test_queue_shows_pending(self, srv_socket):
        with SrvClient(socket_path=srv_socket) as c:
            # Submit a slow job to block the worker
            c.submit(model_type="mock", payload={"duration": 5.0})
            time.sleep(0.2)  # let it start running
            # Submit another that will sit in queue
            c.submit(model_type="mock", payload={"duration": 0.01})
            q = c.queue()
            # Should have at least one active or queued
            assert q["active"] is not None or len(q["queued"]) > 0


# ===========================================================================
# Integration tests — real GPU backends (opt-in via --srv-integration)
# ===========================================================================

pytestmark_srv = pytest.mark.srv_integration


@pytest.fixture()
def srv_full_socket():
    """Start srv with all real backends registered, yield socket path."""
    from indextts_mlx.srv.config import _DEFAULT_MODELS_CONFIG

    sock_fd, socket_path = tempfile.mkstemp(prefix="srv_full_", suffix=".sock", dir="/tmp")
    os.close(sock_fd)
    os.unlink(socket_path)

    server, thread = _start_server(socket_path, _DEFAULT_MODELS_CONFIG, log_level="warning")

    yield socket_path

    server.should_exit = True
    thread.join(timeout=5.0)
    if os.path.exists(socket_path):
        os.unlink(socket_path)


@pytest.mark.srv_integration
class TestIntegrationTTS:
    def test_synthesize_via_client(self, srv_full_socket, tmp_path):
        repo_root = Path(__file__).resolve().parent.parent
        ref_audio = repo_root / "voices" / "british_female.wav"
        assert ref_audio.exists(), f"Reference audio not found: {ref_audio}"

        out_path = str(tmp_path / "out.wav")
        with SrvClient(socket_path=srv_full_socket) as c:
            job = c.synthesize(
                text="Hello world.",
                result_path=out_path,
                spk_audio_prompt=str(ref_audio),
                seed=42,
                use_random=False,
                timeout=120,
                poll_interval=1.0,
            )
            assert job["status"] == "done"
            assert Path(out_path).exists()

            import soundfile as sf
            import numpy as np

            audio, sr = sf.read(out_path)
            assert sr == 22050
            assert len(audio) > 0
            assert np.isfinite(audio).all()


@pytest.mark.srv_integration
class TestIntegrationLLM:
    def test_generate_via_client(self, srv_full_socket):
        with SrvClient(socket_path=srv_full_socket) as c:
            job = c.generate(
                prompt="What is 2+2? Answer with just the number.",
                max_tokens=32,
                timeout=120,
                poll_interval=1.0,
            )
            assert job["status"] == "done"
            assert "text" in job["result"]
            assert len(job["result"]["text"]) > 0


@pytest.mark.srv_integration
class TestIntegrationWhisperX:
    def test_transcribe_via_client(self, srv_full_socket, tmp_path):
        import numpy as np
        import soundfile as sf

        # Create a short silent audio file
        audio = np.zeros(16000, dtype=np.float32)
        audio_path = str(tmp_path / "silence.wav")
        sf.write(audio_path, audio, 16000)

        with SrvClient(socket_path=srv_full_socket) as c:
            job = c.transcribe(
                audio_path=audio_path,
                language="en",
                timeout=120,
                poll_interval=1.0,
            )
            assert job["status"] == "done"
            assert "text" in job["result"]


@pytest.mark.srv_integration
class TestIntegrationTranslation:
    def test_translate_via_client(self, srv_full_socket):
        with SrvClient(socket_path=srv_full_socket) as c:
            job = c.translate(
                text="Buongiorno, come stai?",
                src_lang="ita",
                tgt_lang="eng",
                timeout=120,
                poll_interval=1.0,
            )
            assert job["status"] == "done"
            assert "text" in job["result"]
            assert len(job["result"]["text"]) > 0


@pytest.mark.srv_integration
class TestIntegrationQwen3VoiceDesign:
    def test_design_voice_via_client(self, srv_full_socket, tmp_path):
        out_path = str(tmp_path / "design.wav")
        with SrvClient(socket_path=srv_full_socket) as c:
            job = c.design_voice(
                text="La lluvia caia suavemente sobre los tejados.",
                instruct="A calm Univision news broadcaster",
                result_path=out_path,
                language="spanish",
                timeout=180,
                poll_interval=1.0,
            )
            assert job["status"] == "done"
            assert Path(out_path).exists()

            import soundfile as sf
            import numpy as np

            audio, sr = sf.read(out_path)
            assert len(audio) > 0
            assert np.isfinite(audio).all()


@pytest.mark.srv_integration
class TestIntegrationQwen3CustomVoice:
    def test_custom_voice_via_client(self, srv_full_socket, tmp_path):
        out_path = str(tmp_path / "custom.wav")
        with SrvClient(socket_path=srv_full_socket) as c:
            job = c.custom_voice(
                text="I am so excited about this!",
                speaker="Vivian",
                result_path=out_path,
                instruct="Very happy and excited.",
                language="english",
                timeout=180,
                poll_interval=1.0,
            )
            assert job["status"] == "done"
            assert Path(out_path).exists()

            import soundfile as sf
            import numpy as np

            audio, sr = sf.read(out_path)
            assert len(audio) > 0
            assert np.isfinite(audio).all()


@pytest.mark.srv_integration
class TestIntegrationQwen3VoiceCloning:
    def test_clone_voice_via_client(self, srv_full_socket, tmp_path):
        repo_root = Path(__file__).resolve().parent.parent
        ref_audio = repo_root / "voices" / "british_female.wav"
        assert ref_audio.exists(), f"Reference audio not found: {ref_audio}"

        out_path = str(tmp_path / "clone.wav")
        with SrvClient(socket_path=srv_full_socket) as c:
            job = c.clone_voice(
                text="Hello world, this is a cloning test.",
                ref_audio=str(ref_audio),
                result_path=out_path,
                ref_text="This is a reference recording.",
                language="english",
                timeout=180,
                poll_interval=1.0,
            )
            assert job["status"] == "done"
            assert Path(out_path).exists()

            import soundfile as sf
            import numpy as np

            audio, sr = sf.read(out_path)
            assert len(audio) > 0
            assert np.isfinite(audio).all()


@pytest.mark.srv_integration
class TestIntegrationCloneVoiceIndextts:
    def test_clone_voice_indextts_engine(self, srv_full_socket, tmp_path):
        repo_root = Path(__file__).resolve().parent.parent
        ref_audio = repo_root / "voices" / "british_female.wav"
        assert ref_audio.exists(), f"Reference audio not found: {ref_audio}"

        out_path = str(tmp_path / "clone_indextts.wav")
        with SrvClient(socket_path=srv_full_socket) as c:
            job = c.clone_voice(
                text="Hello world.",
                ref_audio=str(ref_audio),
                result_path=out_path,
                engine="indextts",
                seed=42,
                use_random=False,
                timeout=120,
                poll_interval=1.0,
            )
            assert job["status"] == "done"
            assert Path(out_path).exists()

            import soundfile as sf
            import numpy as np

            audio, sr = sf.read(out_path)
            assert sr == 22050
            assert len(audio) > 0
            assert np.isfinite(audio).all()
