"""
Gap-aware alignment of two chroma sequences.

A Needleman-Wunsch style alignment with the tempo steps of DTW
(Grachten et al., "Automatic alignment of music performances with
structural differences", ISMIR 2013). A matched frame pair earns its
cosine similarity minus an offset, so unrelated frames lose score, and
skipping a frame on either side costs a fixed penalty. Silent frames
never match and are skipped for free.
"""

import numba
import numpy as np

_NO_MATCH = np.float32(-1e9)

STEP_MATCH = 1
STEP_MATCH_PIANO2 = 2
STEP_MATCH_ORIGINAL2 = 3
STEP_SKIP_PIANO = 4
STEP_SKIP_ORIGINAL = 5


@numba.njit(cache=True)
def _reward(piano, original, piano_silent, original_silent, p, o, offset):
    if piano_silent[p] or original_silent[o]:
        return _NO_MATCH
    dot = np.float32(0.0)
    for k in range(12):
        dot += piano[p, k] * original[o, k]
    return dot - offset


@numba.njit(cache=True)
def end_scores(
    piano, original, piano_silent, original_silent, offset, gap, free_start
):
    """
    Score the alignment of the original against every piano prefix.

    Args:
        piano: Chroma of the piano side, shape `(n_piano, 12)`.
        original: Chroma of the original, shape `(n_original, 12)`.
        piano_silent: Silence flag per piano frame.
        original_silent: Silence flag per original frame.
        offset: Similarity at which a matched pair earns zero.
        gap: Penalty for skipping one non-silent frame.
        free_start: Skip leading piano frames at no cost.

    Returns:
        Array of length `n_piano + 1`. Entry `p` is the best score of
        aligning the whole original with piano frames `[0, p)`.
    """
    n_piano, n_original = piano.shape[0], original.shape[0]
    older = np.zeros(n_original + 1, dtype=np.float32)
    old = np.zeros(n_original + 1, dtype=np.float32)
    new = np.zeros(n_original + 1, dtype=np.float32)
    for o in range(1, n_original + 1):
        old[o] = old[o - 1] - (0.0 if original_silent[o - 1] else gap)
    result = np.empty(n_piano + 1, dtype=np.float32)
    result[0] = old[n_original]
    for p in range(1, n_piano + 1):
        gap_piano = 0.0 if piano_silent[p - 1] else gap
        new[0] = 0.0 if free_start else old[0] - gap_piano
        for o in range(1, n_original + 1):
            r = _reward(
                piano, original, piano_silent, original_silent,
                p - 1, o - 1, offset,
            )  # fmt: skip
            best = old[o - 1] + r
            if p > 1 and older[o - 1] + r > best:
                best = older[o - 1] + r
            if o > 1 and old[o - 2] + r > best:
                best = old[o - 2] + r
            if old[o] - gap_piano > best:
                best = old[o] - gap_piano
            skip = new[o - 1] - (0.0 if original_silent[o - 1] else gap)
            if skip > best:
                best = skip
            new[o] = best
        result[p] = new[n_original]
        older, old, new = old, new, older
    return result


@numba.njit(cache=True)
def align(piano, original, piano_silent, original_silent, offset, gap):
    """
    Align all of `piano` with all of `original` and return the path.

    Args:
        piano: Chroma of the piano side, shape `(n_piano, 12)`.
        original: Chroma of the original, shape `(n_original, 12)`.
        piano_silent: Silence flag per piano frame.
        original_silent: Silence flag per original frame.
        offset: Similarity at which a matched pair earns zero.
        gap: Penalty for skipping one non-silent frame.

    Returns:
        A tuple `(score, piano_match, original_match)`. `piano_match[p]`
        is the original frame matched to piano frame `p`, or -1 when
        the frame was skipped. `original_match` is the reverse map.
    """
    n_piano, n_original = piano.shape[0], original.shape[0]
    score = np.zeros((n_piano + 1, n_original + 1), dtype=np.float32)
    step = np.zeros((n_piano + 1, n_original + 1), dtype=np.uint8)
    for o in range(1, n_original + 1):
        cost = 0.0 if original_silent[o - 1] else gap
        score[0, o] = score[0, o - 1] - cost
        step[0, o] = STEP_SKIP_ORIGINAL
    for p in range(1, n_piano + 1):
        gap_piano = 0.0 if piano_silent[p - 1] else gap
        score[p, 0] = score[p - 1, 0] - gap_piano
        step[p, 0] = STEP_SKIP_PIANO
        for o in range(1, n_original + 1):
            r = _reward(
                piano, original, piano_silent, original_silent,
                p - 1, o - 1, offset,
            )  # fmt: skip
            best = score[p - 1, o - 1] + r
            kind = STEP_MATCH
            if p > 1 and score[p - 2, o - 1] + r > best:
                best = score[p - 2, o - 1] + r
                kind = STEP_MATCH_PIANO2
            if o > 1 and score[p - 1, o - 2] + r > best:
                best = score[p - 1, o - 2] + r
                kind = STEP_MATCH_ORIGINAL2
            if score[p - 1, o] - gap_piano > best:
                best = score[p - 1, o] - gap_piano
                kind = STEP_SKIP_PIANO
            skip = score[p, o - 1] - (0.0 if original_silent[o - 1] else gap)
            if skip > best:
                best = skip
                kind = STEP_SKIP_ORIGINAL
            score[p, o] = best
            step[p, o] = kind

    piano_match = np.full(n_piano, -1, dtype=np.int32)
    original_match = np.full(n_original, -1, dtype=np.int32)
    p, o = n_piano, n_original
    while p > 0 or o > 0:
        kind = step[p, o]
        if kind == STEP_SKIP_PIANO:
            p -= 1
        elif kind == STEP_SKIP_ORIGINAL:
            o -= 1
        else:
            piano_match[p - 1] = o - 1
            original_match[o - 1] = p - 1
            # The frame passed over by a double step shares the match.
            if kind == STEP_MATCH_PIANO2:
                piano_match[p - 2] = o - 1
                p -= 2
                o -= 1
            elif kind == STEP_MATCH_ORIGINAL2:
                original_match[o - 2] = p - 1
                p -= 1
                o -= 2
            else:
                p -= 1
                o -= 1
    return score[n_piano, n_original], piano_match, original_match
