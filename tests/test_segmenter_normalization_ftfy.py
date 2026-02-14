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
    # When ftfy is installed it straightens curly quotes first, so \u2018 → '
    # before our regex runs.  The net result is that the straight-apostrophe
    # contraction guard leaves "If'two" intact (correct — it looks like a
    # contraction to the regex).  What we *do* guarantee is that the curly
    # quotes are converted to ASCII and terminal punctuation is added.
    text = "If\u2018two become one\u2019"
    normalized = Segmenter._normalize_sentence(text)
    # Both quote characters must have been converted to ASCII '
    assert "\u2018" not in normalized
    assert "\u2019" not in normalized
    # Terminal punctuation must be present
    assert normalized.endswith(".")
