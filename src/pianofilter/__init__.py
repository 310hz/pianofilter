"""Check and trim solo piano covers against the original song."""

from importlib.metadata import PackageNotFoundError, version

from pianofilter.analyze import Analysis, Config, Unmatched, analyze
from pianofilter.audio import load_audio
from pianofilter.features import (
    Features,
    extract_features,
    features_from_samples,
)
from pianofilter.judge import Criteria, judge
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
