"""Shared speaker-candidate ranking.

match.py, whospoke.py and (later) calibrate_threshold.py all need to answer the
same question — "given this cluster's embedding, who are the most likely speakers
and how sure are we?" — so the logic lives here once. Three copies of a cosine
lambda is how the CLI and the dashboard start disagreeing about what "confident"
means.

The z-score is the important part. Raw cosine is not comparable across clusters:
enrollment clips are close-mic phone audio while lecture clusters are far-field
room audio, and that cross-channel gap compresses every score downward. A
threshold derived from enrollment-vs-enrollment similarity (same channel) applied
to cluster-vs-enrollment scores (cross channel) rejected two *correct* matches on
2026-08-31 — Kaiting at 0.340 and Rija at 0.225.

Normalising each cluster against its own cohort (AS-Norm style) removes that
offset. On the eight verified decisions it separated cleanly:

    enrolled & correct   z 2.20 – 8.56
    unenrolled speaker   z 1.97

That is ONE negative example, so z ranks review priority — it does not decide
identity. See the plan's "Deferred" section.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

# Below this many other candidates the cohort statistics are meaningless.
MIN_COHORT = 3


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def cluster_zscore(sims: dict[str, float], name: str) -> float | None:
    """How far `name` stands out from the rest of this cluster's candidates.

    The cohort is every OTHER candidate for this same cluster, so the channel
    offset that shifts all of them together cancels out.
    """
    others = np.array([s for n, s in sims.items() if n != name], dtype=np.float64)
    if others.size < MIN_COHORT:
        return None
    sd = float(others.std())
    if sd < 1e-6:
        return None
    return round((sims[name] - float(others.mean())) / sd, 3)


def rank_candidates(emb: np.ndarray, voiceprints: dict[str, np.ndarray],
                    top_n: int = 5) -> list[dict]:
    """Ranked candidates for one cluster: [{name, sim, z}, ...] best first."""
    sims = {n: cosine(emb, ref) for n, ref in voiceprints.items()}
    ordered = sorted(sims, key=sims.get, reverse=True)[:top_n]
    return [
        {"name": n, "sim": round(sims[n], 3), "z": cluster_zscore(sims, n)}
        for n in ordered
    ]


def decide(candidates: list[dict], policy: dict) -> str:
    """Classify a cluster's top candidate as accepted / review / unknown.

    'review' means "a human should look at this", not "this is wrong" — the
    Kaiting and Rija cases both scored inside the review band and were correct.
    """
    if not candidates:
        return "unknown"
    top = candidates[0]
    sim, z = top["sim"], top.get("z")
    if sim >= policy.get("accept_sim", 0.45) and (z is None or z >= policy.get("accept_z", 3.0)):
        return "accepted"
    if sim >= policy.get("review_sim_floor", 0.20):
        return "review"
    return "unknown"


def voiceprints_hash(voiceprints: dict) -> str:
    """Identity of the voiceprint set a decision was made against.

    A confirmation is only evidence about the voiceprints that produced it, so
    every stored record carries this. Without it, calibration silently pools
    scores from before and after a learning run.
    """
    payload = json.dumps({k: list(np.asarray(v).round(6)) for k, v in sorted(voiceprints.items())})
    return "sha1:" + hashlib.sha1(payload.encode()).hexdigest()[:16]


def load_policy(config: dict) -> dict:
    """review_policy from config.json, with defaults set from observed data.

    Defaults are deliberately below config's legacy `match_threshold` of 0.7 —
    every verified correct match fell in 0.225-0.831, so a 0.7 gate would send
    almost everything to review.
    """
    policy = {
        "accept_sim": 0.45,
        "review_sim_floor": 0.20,
        "accept_z": 3.0,
        "audit_rate": 0.10,
        "top_n_candidates": 5,
    }
    policy.update(config.get("review_policy", {}))
    return policy


def audit_sampled(cluster_id: str, audit_rate: float) -> bool:
    """Deterministically flag a fraction of *confident* clusters for review too.

    Without this the ground-truth store only ever contains hard cases, and any
    threshold later fitted to it is unrepairably optimistic — there would be no
    high-similarity negatives in the sample. Deterministic (not random) so a
    page reload doesn't reshuffle which clusters ask for review.
    """
    if audit_rate <= 0:
        return False
    digest = hashlib.sha1(cluster_id.encode()).hexdigest()
    return (int(digest[:8], 16) % 10_000) < audit_rate * 10_000
