"""Stable ID alignment and historical strict-greater confidence selection."""

import math

from selectvp.utils.io import index_by_id

BRANCHES = ("bl", "es", "od", "km")


def select_candidate(candidates):
    if not candidates:
        raise ValueError("No candidates")
    for candidate in candidates:
        score = candidate.get("conf")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"Non-finite/missing confidence: {candidate.get('id')}")
    # Historical behavior: even failed candidates have score 0 and participate.
    # A score-0 failure may outrank a valid negative score; retain and report it.
    best = candidates[0]
    for candidate in candidates[1:]:
        if candidate["conf"] > best["conf"]:
            best = candidate
    return dict(best)


def select_aligned(branch_results, expected_ids=None):
    if tuple(branch_results) != BRANCHES:
        raise ValueError(f"Expected branch order {BRANCHES}")
    indexes = {branch: index_by_id(rows) for branch, rows in branch_results.items()}
    keys = list(indexes["bl"])
    if expected_ids is not None:
        keys = list(index_by_id([{"id": x} for x in expected_ids]))
    for branch, index in indexes.items():
        if set(index) != set(keys):
            raise ValueError(f"ID set mismatch in {branch}: missing={len(set(keys)-set(index))}, extra={len(set(index)-set(keys))}")
    result = []
    for key in keys:
        candidates = [dict(indexes[b][key], branch=b) for b in BRANCHES]
        best = select_candidate(candidates)
        best["selected_branch"] = best["branch"]
        best["candidate_scores"] = {c["branch"]: c["conf"] for c in candidates}
        result.append(best)
    return result
