"""Compare a piano cover with its original song."""

from dataclasses import asdict, dataclass

import numpy as np

from pianofilter.align import align, end_scores
from pianofilter.features import (
    CHROMA_SECONDS,
    RMS_SECONDS,
    WINDOW_SECONDS,
    Features,
)


@dataclass(frozen=True)
class Config:
    """
    Parameters of the comparison.

    Attributes:
        similarity_offset: Chroma cosine similarity at which a matched
            frame pair earns zero. Pairs below it lose score.
        gap_penalty: Score lost per skipped non-silent frame.
        silence_db: Level below which a frame is silent, relative to
            the loud level of the file (its 95th percentile).
        min_silence_seconds: Shortest silence that splits the piano
            recording into blocks.
        piano_threshold: Solo piano score below which a window counts
            as not piano.
        min_low_run_seconds: Shortest run of low piano scores whose
            edges also split the recording into blocks.
        max_blocks: Upper bound on the number of blocks.
        min_length_ratio: Smallest allowed length of a candidate
            region relative to the original.
        max_length_ratio: Largest allowed length of a candidate region
            relative to the original.
        pad_seconds: Silence kept around the chosen region.
    """

    similarity_offset: float = 0.8
    gap_penalty: float = 0.05
    silence_db: float = -40.0
    min_silence_seconds: float = 0.7
    piano_threshold: float = 0.6
    min_low_run_seconds: float = 8.0
    max_blocks: int = 10
    min_length_ratio: float = 0.6
    max_length_ratio: float = 1.6
    pad_seconds: float = 0.5


@dataclass(frozen=True)
class Unmatched:
    """
    Seconds of non-silent audio left without a counterpart.

    Attributes:
        head: At the beginning.
        tail: At the end.
        inner_total: Between the first and the last match, in total.
        inner_longest: Longest single stretch between matches.
    """

    head: float
    tail: float
    inner_total: float
    inner_longest: float


@dataclass(frozen=True)
class Analysis:
    """
    Result of comparing one piano recording with its original.

    Attributes:
        start: Start of the performance in the piano file, in seconds.
            `None` when no region could be chosen.
        end: End of the performance in the piano file, in seconds.
        shift: Semitones by which the piano is transposed relative to
            the original, in `[-5, 6]`.
        shift_margin: Score lead of `shift` over the runner-up, per
            second of the original. Small values mean an unsure key.
        score: Alignment score of the chosen region.
        coverage: Fraction of the original that found a counterpart.
        similarity: Mean chroma similarity of the matched frames.
        tempo_ratio: Piano time per original time over the matched
            span.
        original_unmatched: Parts of the original the piano skips.
        piano_unmatched: Parts of the region absent from the original.
        piano_score_mean: Mean solo piano score inside the region.
        piano_low_fraction: Fraction of the region scored as not piano.
        piano_low_longest: Longest stretch of the region scored as not
            piano, in seconds.
        original_piano_fraction: Fraction of the original scored as
            solo piano.
        n_blocks: Number of blocks the piano file was split into.
    """

    start: float | None
    end: float | None
    shift: int
    shift_margin: float
    score: float
    coverage: float
    similarity: float
    tempo_ratio: float
    original_unmatched: Unmatched
    piano_unmatched: Unmatched
    piano_score_mean: float | None
    piano_low_fraction: float | None
    piano_low_longest: float | None
    original_piano_fraction: float | None
    n_blocks: int

    def to_dict(self) -> dict:
        """Return the analysis as plain JSON-serializable values."""
        return asdict(self)


@dataclass(frozen=True)
class _Block:
    frame_start: int
    frame_end: int
    cut_start: float
    cut_end: float


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def _silent(rms_db: np.ndarray, config: Config) -> np.ndarray:
    if len(rms_db) == 0:
        return np.zeros(0, dtype=bool)
    loud = float(np.percentile(rms_db, 95))
    return rms_db < loud + config.silence_db


def _chroma_silent(silent: np.ndarray, n_frames: int) -> np.ndarray:
    step = round(CHROMA_SECONDS / RMS_SECONDS)
    padded = np.ones(n_frames * step, dtype=bool)
    n = min(len(silent), len(padded))
    padded[:n] = silent[:n]
    return padded.reshape(n_frames, step).all(axis=1)


def _gaps(piano: Features, silent: np.ndarray, config: Config) -> list:
    """Return the `(start, end)` seconds that separate blocks."""
    silences = [
        (a * RMS_SECONDS, b * RMS_SECONDS)
        for a, b in _runs(silent)
        if (b - a) * RMS_SECONDS >= config.min_silence_seconds
    ]
    edges = []
    if piano.piano_scores is not None:
        low = piano.piano_scores < config.piano_threshold
        for a, b in _runs(low):
            if b - a >= config.min_low_run_seconds:
                # Window i is centred half a window after second i.
                for t in (a + WINDOW_SECONDS / 2, b - 1 + WINDOW_SECONDS / 2):
                    if 0 < t < piano.duration:
                        edges.append((float(t), float(t)))
    n_keep = max(config.max_blocks - 1 - len(edges), 0)
    silences = sorted(silences, key=lambda g: g[0] - g[1])[:n_keep]
    return sorted(silences + edges)


def _blocks(piano: Features, silent: np.ndarray, config: Config) -> list:
    gaps = _gaps(piano, silent, config)
    pad = config.pad_seconds
    bounds = [(0.0, 0.0), *gaps, (piano.duration, piano.duration)]
    blocks = []
    for (a0, b0), (a1, b1) in zip(bounds[:-1], bounds[1:], strict=True):
        if a1 - b0 < 1.0:
            continue
        blocks.append(
            _Block(
                frame_start=int(b0 / CHROMA_SECONDS),
                frame_end=min(
                    int(np.ceil(a1 / CHROMA_SECONDS)), len(piano.chroma)
                ),
                cut_start=max((a0 + b0) / 2, b0 - pad),
                cut_end=min((a1 + b1) / 2, a1 + pad),
            )
        )
    return blocks


def _unmatched(matched: np.ndarray, silent: np.ndarray) -> Unmatched:
    head = tail = total = longest = 0.0
    for a, b in _runs(~matched):
        seconds = float((~silent[a:b]).sum() * CHROMA_SECONDS)
        if a == 0:
            head = seconds
        elif b == len(matched):
            tail = seconds
        else:
            total += seconds
            longest = max(longest, seconds)
    return Unmatched(head, tail, total, longest)


def _best_shift(piano, original, piano_silent, original_silent, config):
    scores = np.array([
        end_scores(
            piano,
            np.roll(original, shift, axis=1),
            piano_silent,
            original_silent,
            np.float32(config.similarity_offset),
            np.float32(config.gap_penalty),
            True,
        ).max()
        for shift in range(12)
    ])  # fmt: skip
    order = np.argsort(scores)
    seconds = max(float((~original_silent).sum()) * CHROMA_SECONDS, 1.0)
    margin = float(scores[order[-1]] - scores[order[-2]]) / seconds
    return int(order[-1]), margin


def _best_region(
    blocks, piano, original, piano_silent, original_silent, config
):
    """Pick the run of consecutive blocks that aligns best."""
    active = np.flatnonzero(~original_silent)
    seconds_original = (active[-1] - active[0] + 1) * CHROMA_SECONDS
    candidates = []
    for i, first in enumerate(blocks):
        scores = end_scores(
            piano[first.frame_start :],
            original,
            piano_silent[first.frame_start :],
            original_silent,
            np.float32(config.similarity_offset),
            np.float32(config.gap_penalty),
            False,
        )
        for j in range(i, len(blocks)):
            last = blocks[j]
            n_frames = last.frame_end - first.frame_start
            ratio = n_frames * CHROMA_SECONDS / seconds_original
            in_range = (
                config.min_length_ratio <= ratio <= config.max_length_ratio
            )
            candidates.append((in_range, float(scores[n_frames]), i, j))
    # Fall back to every candidate when none has a plausible length, so
    # that the caller still gets measurements to reject on.
    _, score, i, j = max(candidates)
    return score, blocks[i], blocks[j]


def _piano_stats(piano, start, end, config):
    if piano.piano_scores is None:
        return None, None, None
    centres = np.arange(len(piano.piano_scores)) + WINDOW_SECONDS / 2
    scores = piano.piano_scores[(centres >= start) & (centres <= end)]
    if len(scores) == 0:
        return None, None, None
    low = scores < config.piano_threshold
    longest = max((b - a for a, b in _runs(low)), default=0)
    return float(scores.mean()), float(low.mean()), float(longest)


def analyze(
    original: Features, piano: Features, config: Config | None = None
) -> Analysis:
    """
    Find the performance in a piano file and compare it.

    The piano file is split into blocks at silences and at the edges of
    long stretches that do not sound like piano. Every run of
    consecutive blocks is aligned with the original, and the run with
    the best alignment score becomes the region.

    Args:
        original: Features of the original song.
        piano: Features of the piano cover.
        config: Parameters. Defaults to `Config()`.

    Returns:
        The chosen region and the measurements to judge the pair on.
    """
    config = config or Config()
    piano_silent = _chroma_silent(
        _silent(piano.rms_db, config), len(piano.chroma)
    )
    original_silent = _chroma_silent(
        _silent(original.rms_db, config), len(original.chroma)
    )
    original_piano_fraction = (
        None
        if original.piano_scores is None or len(original.piano_scores) == 0
        else float((original.piano_scores >= config.piano_threshold).mean())
    )
    blocks = _blocks(piano, _silent(piano.rms_db, config), config)
    if not blocks or original_silent.all():
        empty = Unmatched(0.0, 0.0, 0.0, 0.0)
        return Analysis(
            None, None, 0, 0.0, 0.0, 0.0, 0.0, 0.0, empty, empty,
            None, None, None, original_piano_fraction, len(blocks),
        )  # fmt: skip

    shift, margin = _best_shift(
        piano.chroma, original.chroma, piano_silent, original_silent, config
    )
    shifted = np.roll(original.chroma, shift, axis=1)
    score, first, last = _best_region(
        blocks, piano.chroma, shifted, piano_silent, original_silent, config
    )
    frames = slice(first.frame_start, last.frame_end)
    _, piano_match, original_match = align(
        piano.chroma[frames],
        shifted,
        piano_silent[frames],
        original_silent,
        np.float32(config.similarity_offset),
        np.float32(config.gap_penalty),
    )

    matched_piano = np.flatnonzero(piano_match >= 0)
    matched_original = piano_match[matched_piano]
    if len(matched_piano) > 1:
        similarity = float(
            (piano.chroma[frames][matched_piano] * shifted[matched_original])
            .sum(axis=1)
            .mean()
        )
        tempo_ratio = float(
            (matched_piano[-1] - matched_piano[0])
            / max(matched_original[-1] - matched_original[0], 1)
        )
    else:
        similarity = tempo_ratio = 0.0
    n_active = max(int((~original_silent).sum()), 1)
    coverage = float(((original_match >= 0) & ~original_silent).sum())
    mean, low_fraction, low_longest = _piano_stats(
        piano, first.cut_start, last.cut_end, config
    )
    return Analysis(
        start=first.cut_start,
        end=last.cut_end,
        shift=(shift + 5) % 12 - 5,
        shift_margin=margin,
        score=score,
        coverage=coverage / n_active,
        similarity=similarity,
        tempo_ratio=tempo_ratio,
        original_unmatched=_unmatched(original_match >= 0, original_silent),
        piano_unmatched=_unmatched(piano_match >= 0, piano_silent[frames]),
        piano_score_mean=mean,
        piano_low_fraction=low_fraction,
        piano_low_longest=low_longest,
        original_piano_fraction=original_piano_fraction,
        n_blocks=len(blocks),
    )
