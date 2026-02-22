"""Tests for model manager load/unload lifecycle."""

from unittest.mock import patch, MagicMock

from indextts_mlx.srv.backends.mock import MockBackend
from indextts_mlx.srv.model_manager import ModelManager


def _make_config(**extra_backends):
    config = {
        "backends": {
            "mock": {
                "default": "default",
                "models": {
                    "default": {},
                    "variant_a": {"param": "a"},
                    "variant_b": {"param": "b"},
                },
            },
            **extra_backends,
        },
    }
    return config


def test_load_unloads_previous():
    mgr = ModelManager(_make_config())
    mgr.register("mock", MockBackend)

    backend_a = mgr.ensure_loaded("mock", "variant_a")
    assert backend_a._loaded

    backend_b = mgr.ensure_loaded("mock", "variant_b")
    assert backend_b._loaded
    # Previous backend should have been unloaded
    assert not backend_a._loaded


def test_same_model_reused():
    mgr = ModelManager(_make_config())
    mgr.register("mock", MockBackend)

    backend_a = mgr.ensure_loaded("mock", "variant_a")
    backend_a2 = mgr.ensure_loaded("mock", "variant_a")
    assert backend_a is backend_a2


def test_different_name_same_type_reloads():
    mgr = ModelManager(_make_config())
    mgr.register("mock", MockBackend)

    backend_a = mgr.ensure_loaded("mock", "variant_a")
    assert mgr.current_key == ("mock", "variant_a")

    backend_b = mgr.ensure_loaded("mock", "variant_b")
    assert mgr.current_key == ("mock", "variant_b")
    assert backend_a is not backend_b
    assert not backend_a._loaded


def test_unload_calls_cleanup():
    mgr = ModelManager(_make_config())
    mgr.register("mock", MockBackend)
    mgr.ensure_loaded("mock", "default")

    with patch("gc.collect") as mock_gc:
        mgr.unload()
        mock_gc.assert_called_once()

    assert mgr.current_key is None


def test_status_reflects_loaded():
    mgr = ModelManager(_make_config())
    mgr.register("mock", MockBackend)

    assert mgr.status() == {"loaded_model": None}

    mgr.ensure_loaded("mock", "variant_a")
    assert mgr.status() == {"loaded_model": {"type": "mock", "name": "variant_a"}}

    mgr.unload()
    assert mgr.status() == {"loaded_model": None}


def test_default_model_resolved():
    mgr = ModelManager(_make_config())
    mgr.register("mock", MockBackend)

    backend = mgr.ensure_loaded("mock")  # no model_name → uses default
    assert mgr.current_key == ("mock", "default")
