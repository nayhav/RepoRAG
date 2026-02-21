"""Reciprocal Rank Fusion and score normalization utilities."""

from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    For each document *d* that appears in any list, the fused score is:

        score(d) = Σ  1 / (k + rank_i(d))

    where rank_i(d) is the 1-based rank of *d* in list *i*.

    Parameters
    ----------
    ranked_lists:
        Each inner list contains ``(item_id, score)`` tuples already sorted
        by score descending.
    k:
        The RRF constant (default 60).  Higher values dampen the influence
        of high-ranked items.

    Returns
    -------
    list[tuple[str, float]]
        Merged list sorted by fused score descending.
    """
    fused_scores: dict[str, float] = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, (item_id, _score) in enumerate(ranked_list, start=1):
            fused_scores[item_id] += 1.0 / (k + rank)

    results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    logger.debug(
        "RRF fused %d unique items from %d ranked lists",
        len(results),
        len(ranked_lists),
    )
    return results


def normalize_scores(
    results: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Min-max normalize scores to the [0, 1] range.

    If all scores are equal (or the list has ≤1 element), every score is
    set to 1.0 so downstream consumers can still compare.
    """
    if not results:
        return []

    scores = [s for _, s in results]
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score

    if score_range == 0.0:
        return [(item_id, 1.0) for item_id, _ in results]

    return [
        (item_id, (score - min_score) / score_range)
        for item_id, score in results
    ]
