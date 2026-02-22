"""GPU resource management service for serialized model access."""

from indextts_mlx.srv.config import SrvConfig, load_models_config
from indextts_mlx.srv.app import create_app
from indextts_mlx.srv.client import SrvClient

__all__ = ["create_app", "SrvClient", "SrvConfig", "load_models_config"]
