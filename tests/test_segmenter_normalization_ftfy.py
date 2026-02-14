import importlib.util
import sys
from pathlib import Path


def _load_segmenter():
    path = Path(__file__).resolve().parent.parent / "indextts_mlx" / "segmenter.py"
    module_name = "indextts_mlx.segmenter_raw"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.Segmenter


def test_normalize_sentence_fix_text_applies_spacing_and_quotes():
    Segmenter = _load_segmenter()
    text = "If‘two become one’"
    normalized = Segmenter._normalize_sentence(text)
    assert normalized == "If 'two become one'."
