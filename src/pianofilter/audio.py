"""Audio decoding."""

import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 22050


def load_audio(
    fpath_audio: Path, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """
    Decode an audio file to mono float32 with ffmpeg.

    Args:
        fpath_audio: Any file ffmpeg can read.
        sample_rate: Sampling rate of the returned signal.

    Returns:
        The samples, shape `(n,)`.

    Raises:
        RuntimeError: If ffmpeg fails or the file has no audio.
    """
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-nostdin",
            "-i", str(fpath_audio),
            "-vn", "-ac", "1", "-ar", str(sample_rate),
            "-f", "f32le", "-",
        ],
        capture_output=True,
        check=False,
    )  # fmt: skip
    if result.returncode != 0 or not result.stdout:
        message = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg could not decode {fpath_audio}: {message}")
    return np.frombuffer(result.stdout, dtype=np.float32)
