"""Per-file measurements that the comparison works on."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import librosa
import numpy as np

from pianofilter.audio import SAMPLE_RATE, load_audio

if TYPE_CHECKING:
    from pianofilter.piano import PianoScorer

_HOP_LENGTH = 512
_CHROMA_STEP = 8
_CENS_SMOOTH = 41

RMS_SECONDS = _HOP_LENGTH / SAMPLE_RATE
CHROMA_SECONDS = RMS_SECONDS * _CHROMA_STEP
WINDOW_SECONDS = 5.0


@dataclass(frozen=True)
class Features:
    """
    Measurements of one audio file.

    Attributes:
        duration: Length in seconds.
        chroma: CENS chroma, shape `(n, 12)`, one row per
            `CHROMA_SECONDS`. Rows have unit length.
        rms_db: Level in dBFS, one value per `RMS_SECONDS`.
        piano_scores: Solo piano score of each 5-second window (window
            `i` starts at second `i`), or `None` when not computed.
    """

    duration: float
    chroma: np.ndarray
    rms_db: np.ndarray
    piano_scores: np.ndarray | None = None

    def save(self, fpath: Path) -> None:
        """
        Write the features to an `.npz` file.

        Args:
            fpath: Destination path.
        """
        arrays: dict[str, np.ndarray] = {
            "duration": np.array(self.duration),
            "chroma": self.chroma,
            "rms_db": self.rms_db,
        }
        if self.piano_scores is not None:
            arrays["piano_scores"] = self.piano_scores
        np.savez_compressed(fpath, allow_pickle=False, **arrays)

    @classmethod
    def load(cls, fpath: Path) -> "Features":
        """
        Read features written by `save`.

        Args:
            fpath: Path of the `.npz` file.

        Returns:
            The stored features.
        """
        with np.load(fpath) as data:
            return cls(
                duration=float(data["duration"]),
                chroma=data["chroma"],
                rms_db=data["rms_db"],
                piano_scores=(
                    data["piano_scores"] if "piano_scores" in data else None
                ),
            )


def features_from_samples(
    samples: np.ndarray, scorer: "PianoScorer | None" = None
) -> Features:
    """
    Compute features from a decoded signal.

    Args:
        samples: Mono signal at 22,050 Hz, as from `load_audio`.
        scorer: Solo piano classifier. Piano scores are skipped when
            omitted.

    Returns:
        The features of the signal.
    """
    tuning = librosa.estimate_tuning(y=samples, sr=SAMPLE_RATE)
    chroma = librosa.feature.chroma_cens(
        y=samples,
        sr=SAMPLE_RATE,
        hop_length=_HOP_LENGTH,
        tuning=tuning,
        win_len_smooth=_CENS_SMOOTH,
    )
    rms = librosa.feature.rms(y=samples, hop_length=_HOP_LENGTH)[0]
    return Features(
        duration=len(samples) / SAMPLE_RATE,
        chroma=chroma[:, ::_CHROMA_STEP].T.astype(np.float32),
        rms_db=(20 * np.log10(np.maximum(rms, 1e-5))).astype(np.float32),
        piano_scores=None if scorer is None else scorer.score(samples),
    )


def extract_features(
    fpath_audio: Path, scorer: "PianoScorer | None" = None
) -> Features:
    """
    Decode an audio file and compute its features.

    Args:
        fpath_audio: Any file ffmpeg can read.
        scorer: Solo piano classifier. Piano scores are skipped when
            omitted.

    Returns:
        The features of the file.
    """
    return features_from_samples(load_audio(fpath_audio), scorer)
