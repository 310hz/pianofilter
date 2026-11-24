"""Check and trim solo piano covers against the original song."""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from pianofilter.analyze import Analysis, Config, Unmatched, analyze
from pianofilter.audio import load_audio
from pianofilter.features import (
    Features,
    extract_features,
    features_from_samples,
)
from pianofilter.judge import Criteria, judge

if TYPE_CHECKING:
    from pianofilter.piano import PianoScorer, download_weights

try:
    __version__ = version("pianofilter")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "Analysis",
    "Config",
    "Criteria",
    "Features",
    "PianoScorer",
    "Unmatched",
    "analyze",
    "download_weights",
    "extract_features",
    "features_from_samples",
    "judge",
    "load_audio",
]


def __getattr__(name: str) -> object:
    # torch is imported only when the piano scorer is asked for, so
    # that worker processes computing the other features stay light.
    if name in ("PianoScorer", "download_weights"):
        from pianofilter import piano

        return getattr(piano, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
