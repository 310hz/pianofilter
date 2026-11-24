"""
Solo piano scoring with the Aria-CL classifier weights.

The network and the mel front end are reimplemented here so that the
published checkpoint of Aria-CL (https://github.com/loubbrad/aria-cl)
can be loaded without its package, whose audio loader no longer works
with current torchaudio.
"""

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torchaudio
from safetensors.torch import load_file
from torch import nn

from pianofilter.audio import SAMPLE_RATE, load_audio
from pianofilter.features import WINDOW_SECONDS

WEIGHTS_URL = (
    "https://storage.googleapis.com/aria-checkpoints/cl/large-0.1.safetensors"
)
WEIGHTS_SHA256 = (
    "94d9609d358f9895159fe4f83620bb64286a4f9045fd16079b4feaca984e2a75"
)
HOP_SECONDS = 1.0

_N_FFT = 2048
_HOP_LENGTH = 220
_N_MELS = 256
_CHANNELS = (8, 16, 32, 64, 128)
_HIDDEN = 512
_KEY_PREFIX = "_orig_mod."


def default_weights_path() -> Path:
    """Return where the checkpoint is cached."""
    return (
        Path.home()
        / ".cache"
        / "pianofilter"
        / "aria-cl-large-0.1.safetensors"
    )


def download_weights(fpath_weights: Path | None = None) -> Path:
    """
    Download the Aria-CL checkpoint unless it is already there.

    Args:
        fpath_weights: Destination. Defaults to
            `default_weights_path()`.

    Returns:
        The path of the checkpoint.

    Raises:
        RuntimeError: If the downloaded file has an unexpected hash.
    """
    fpath_weights = fpath_weights or default_weights_path()
    if fpath_weights.exists():
        return fpath_weights
    fpath_weights.parent.mkdir(parents=True, exist_ok=True)
    fpath_part = fpath_weights.with_suffix(".part")
    urllib.request.urlretrieve(WEIGHTS_URL, fpath_part)  # noqa: S310
    digest = hashlib.sha256(fpath_part.read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256:
        fpath_part.unlink()
        raise RuntimeError(f"Unexpected checkpoint hash: {digest}")
    fpath_part.rename(fpath_weights)
    return fpath_weights


class _Classifier(nn.Module):
    # Attribute names and Sequential indices follow the checkpoint keys.
    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        n_in = 1
        for i, n_out in enumerate(_CHANNELS):
            last = i == len(_CHANNELS) - 1
            layers += [
                nn.Conv2d(n_in, n_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(n_out),
                nn.ReLU(),
                nn.Identity() if last else nn.MaxPool2d(2),
            ]
            n_in = n_out
        self.conv_layers = nn.Sequential(*layers)
        n_pool = len(_CHANNELS) - 1
        n_time = int(WINDOW_SECONDS * SAMPLE_RATE) // _HOP_LENGTH
        n_flat = _CHANNELS[-1] * (_N_MELS >> n_pool) * (n_time >> n_pool)
        self.mlp_layers = nn.Sequential(
            nn.Linear(n_flat, _HIDDEN), nn.ReLU(), nn.Dropout(0.1)
        )
        self.output_layer = nn.Linear(_HIDDEN, 1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(mel.unsqueeze(1))
        return self.output_layer(self.mlp_layers(x.flatten(1))).squeeze(1)


class PianoScorer:
    """Scores how much each 5-second window sounds like solo piano."""

    def __init__(
        self,
        fpath_weights: Path | None = None,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        """
        Load the classifier.

        Args:
            fpath_weights: Aria-CL `large-0.1` checkpoint. Downloaded
                to the cache when omitted.
            device: Torch device. Defaults to CUDA when available.
            batch_size: Windows per forward pass. A window batch of 64
                needs roughly 0.5 GB of GPU memory.
        """
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size = batch_size
        state = load_file(download_weights(fpath_weights))
        state = {k.removeprefix(_KEY_PREFIX): v for k, v in state.items()}
        self._model = _Classifier()
        self._model.load_state_dict(state)
        self._model.eval().to(self.device)
        self._spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=_N_FFT, hop_length=_HOP_LENGTH
        ).to(self.device)
        self._mel_scale = torchaudio.transforms.MelScale(
            n_mels=_N_MELS,
            sample_rate=SAMPLE_RATE,
            n_stft=_N_FFT // 2 + 1,
            f_min=20,
            f_max=SAMPLE_RATE // 2,
        ).to(self.device)

    def _log_mel(self, windows: torch.Tensor) -> torch.Tensor:
        mel = self._mel_scale(self._spectrogram(windows)[..., :-1])
        log_mel = mel.clamp(min=1e-10).log10()
        peak = log_mel.amax(dim=(1, 2), keepdim=True)
        return (torch.maximum(log_mel, peak - 8.0) + 4.0) / 4.0

    @torch.inference_mode()
    def score(self, samples: np.ndarray) -> np.ndarray:
        """
        Score every window of a signal.

        Args:
            samples: Mono signal at 22,050 Hz, as from `load_audio`.

        Returns:
            Scores in `[0, 1]`, one per window. Window `i` covers
            seconds `[i, i + 5)`. Empty when the signal is shorter
            than one window.
        """
        n_window = int(WINDOW_SECONDS * SAMPLE_RATE)
        n_hop = int(HOP_SECONDS * SAMPLE_RATE)
        if len(samples) < n_window:
            return np.zeros(0, dtype=np.float32)
        windows = torch.from_numpy(samples.copy()).unfold(0, n_window, n_hop)
        scores = []
        for i in range(0, windows.shape[0], self.batch_size):
            batch = windows[i : i + self.batch_size].to(self.device)
            logits = self._model(self._log_mel(batch))
            scores.append(torch.sigmoid(logits).cpu())
        return torch.cat(scores).numpy().astype(np.float32)

    def score_file(self, fpath_audio: Path) -> np.ndarray:
        """
        Decode an audio file and score every window of it.

        Args:
            fpath_audio: Any file ffmpeg can read.

        Returns:
            Scores as from `score`.
        """
        return self.score(load_audio(fpath_audio))
