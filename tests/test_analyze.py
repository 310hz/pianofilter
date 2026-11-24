import numpy as np

from pianofilter import Features, analyze, judge
from pianofilter.features import CHROMA_SECONDS, RMS_SECONDS

STEP = round(CHROMA_SECONDS / RMS_SECONDS)
RNG = np.random.default_rng(0)


def song(n_frames: int) -> np.ndarray:
    """Chroma that changes chord every few frames."""
    chords = RNG.random((n_frames // 10 + 1, 12)) ** 4
    chroma = np.repeat(chords, 10, axis=0)[:n_frames]
    return chroma / np.linalg.norm(chroma, axis=1, keepdims=True)


def noise(n_frames: int) -> np.ndarray:
    chroma = RNG.random((n_frames, 12)) ** 4
    return chroma / np.linalg.norm(chroma, axis=1, keepdims=True)


def features(parts: list[np.ndarray | int]) -> Features:
    """Join chroma parts; an int is that many frames of silence."""
    chroma, rms_db = [], []
    for part in parts:
        if isinstance(part, int):
            chroma.append(np.zeros((part, 12)))
            rms_db.append(np.full(part * STEP, -90.0))
        else:
            chroma.append(part)
            rms_db.append(np.full(len(part) * STEP, -20.0))
    chroma = np.concatenate(chroma).astype(np.float32)
    return Features(
        duration=len(chroma) * CHROMA_SECONDS,
        chroma=chroma,
        rms_db=np.concatenate(rms_db).astype(np.float32),
    )


def seconds(n_frames: int) -> float:
    return n_frames * CHROMA_SECONDS


def test_identical_pair_is_accepted():
    body = song(600)
    result = analyze(features([body]), features([body]))
    assert result.shift == 0
    assert result.coverage > 0.95
    assert judge(result) == []


def test_talk_before_and_after_is_cut_at_the_silences():
    body = song(600)
    piano = features([noise(80), 10, body, 10, noise(120)])
    result = analyze(features([body]), piano)
    assert result.start is not None and result.end is not None
    assert seconds(80) < result.start <= seconds(90)
    assert seconds(690) <= result.end < seconds(700)
    assert judge(result) == []


def test_pause_inside_the_performance_is_kept():
    body = song(600)
    piano = features([noise(80), 10, body[:400], 10, body[400:]])
    result = analyze(features([body]), piano)
    assert result.end is not None
    assert result.end > seconds(690)
    assert result.coverage > 0.95


def test_transposed_cover_reports_the_shift():
    body = song(600)
    piano = features([np.roll(body, 2, axis=1)])
    result = analyze(features([body]), piano)
    assert result.shift == 2
    assert judge(result) == ["key"]


def test_dropped_section_is_reported():
    body = song(900)
    piano = features([np.concatenate([body[:300], body[500:]])])
    result = analyze(features([body]), piano)
    assert abs(result.original_unmatched.inner_longest - seconds(200)) < 3
    assert "missing_section" in judge(result)


def test_slower_cover_is_accepted():
    body = song(600)
    piano = features([np.repeat(body, 5, axis=0)[::4]])
    result = analyze(features([body]), piano)
    assert abs(result.tempo_ratio - 1.25) < 0.05
    assert judge(result) == []


def test_different_song_is_rejected():
    result = analyze(features([song(600)]), features([song(600)]))
    assert judge(result) == ["different_song"]
