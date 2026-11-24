"""Turn an analysis into reasons to reject a pair."""

from dataclasses import dataclass

from pianofilter.analyze import Analysis


@dataclass(frozen=True)
class Criteria:
    """
    Limits an analysis must stay within to be accepted.

    The defaults are placeholders that have not been calibrated.

    Attributes:
        min_coverage: Smallest fraction of the original that must find
            a counterpart for the pair to count as the same song.
        max_original_gap: Longest stretch of the original, in seconds,
            that the piano may skip.
        max_piano_gap: Longest stretch inside the region, in seconds,
            that may be absent from the original.
        max_piano_low: Longest stretch inside the region, in seconds,
            that may be scored as not piano.
        max_piano_low_fraction: Largest fraction of the region that
            may be scored as not piano.
        max_original_piano_fraction: Largest fraction of the original
            that may be scored as solo piano.
    """

    min_coverage: float = 0.5
    max_original_gap: float = 15.0
    max_piano_gap: float = 15.0
    max_piano_low: float = 30.0
    max_piano_low_fraction: float = 0.5
    max_original_piano_fraction: float = 0.8


def judge(analysis: Analysis, criteria: Criteria | None = None) -> list[str]:
    """
    List the reasons to reject a pair.

    Args:
        analysis: Result of `analyze`.
        criteria: Limits to apply. Defaults to `Criteria()`.

    Returns:
        Reason codes, empty when the pair is accepted: `no_region`,
        `different_song`, `key`, `missing_section`, `extra_section`,
        `not_piano_solo`, `original_is_piano`.
    """
    criteria = criteria or Criteria()
    reasons = []
    fraction = analysis.original_piano_fraction
    if (
        fraction is not None
        and fraction > criteria.max_original_piano_fraction
    ):
        reasons.append("original_is_piano")
    if analysis.start is None:
        return [*reasons, "no_region"]
    if analysis.coverage < criteria.min_coverage:
        return [*reasons, "different_song"]
    if analysis.shift != 0:
        reasons.append("key")
    gaps = analysis.original_unmatched
    if (
        max(gaps.head, gaps.tail, gaps.inner_longest)
        > criteria.max_original_gap
    ):
        reasons.append("missing_section")
    if analysis.piano_unmatched.inner_longest > criteria.max_piano_gap:
        reasons.append("extra_section")
    low = analysis.piano_low_longest
    low_fraction = analysis.piano_low_fraction
    if (low is not None and low > criteria.max_piano_low) or (
        low_fraction is not None
        and low_fraction > criteria.max_piano_low_fraction
    ):
        reasons.append("not_piano_solo")
    return reasons
