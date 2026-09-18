"""exp1 Phase 0 — Baseline similarity matrix + impostor floor.

Read-only. Touches nothing outside experiments/exp1-enrollment-augmentation/.

Establishes the number the augmented arms have to beat. "0 matches" is not a
metric — what matters is where each cluster's best score sits relative to the
distribution of scores it gets against speakers it definitely is not.

Reuses the production encoder and the same per-speaker-cluster aggregation as
pipeline/match.py, so the baseline is the real system's behaviour, not a
reimplementation of it.

Writes:
    results/baseline.json          full matrix + per-cluster stats + floor
    results/cluster_embeddings.npz cached cluster embeddings (Phases 2-4 reuse)

Run:  .venv/bin/python experiments/exp1-enrollment-augmentation/scripts/p0_baseline.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent.parent
PROJECT = EXP.parent.parent
sys.path.insert(0, str(PROJECT / "pipeline"))

from common import (  # noqa: E402
    AUDIO_LECTURES, TURNS, VOICEPRINTS, MODELS,
    read_json, pick_device_speechbrain, load_wav_mono16k,
)
from match import cosine, embed_segments  # noqa: E402


def main() -> None:
    cfg = json.load(open(EXP / "config.json"))
    results = EXP / "results"
    results.mkdir(parents=True, exist_ok=True)

    audio = AUDIO_LECTURES / cfg["lecture"]
    if not audio.exists():
        sys.exit(f"Lecture audio not found: {audio}")

    turns = read_json(TURNS)
    voiceprints = {k: np.array(v, dtype=np.float32) for k, v in read_json(VOICEPRINTS).items()}
    names = sorted(voiceprints)

    device = pick_device_speechbrain()
    print(f"Phase 0 — baseline")
    print(f"  audio     : {audio.name}")
    print(f"  turns     : {len(turns)}")
    print(f"  enrolled  : {len(names)}  ({', '.join(names)})")
    print(f"  device    : {device}\n")

    from speechbrain.inference.speaker import EncoderClassifier

    signal, sr = load_wav_mono16k(audio)
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(MODELS / "ecapa"),
        run_opts={"device": device},
    )

    # Same aggregation as match.py: all of a diarized speaker's audio, embedded once.
    segs_by_spk = defaultdict(list)
    for t in turns:
        segs_by_spk[t["speaker_id"]].append((t["start"], t["end"]))

    clusters = sorted(segs_by_spk)
    print(f"  clusters  : {len(clusters)}\n")

    embeddings = {}
    matrix = {}
    for spk in clusters:
        segs = segs_by_spk[spk]
        dur = sum(e - s for s, e in segs)
        emb = embed_segments(encoder, signal, sr, segs)
        embeddings[spk] = emb
        matrix[spk] = {n: cosine(emb, voiceprints[n]) for n in names}
        print(f"  embedded {spk:18s} {len(segs):3d} segs  {dur:6.1f}s")

    # ---- Per-cluster stats -------------------------------------------------
    # top1/top2 margin is the shape that matters: Prof's 0.58 margin is what a
    # real identification looks like. z scores top1 against the same cluster's
    # own non-top scores, so it is robust to a cluster that is broadly similar
    # to everyone (a sign of a degenerate embedding).
    per_cluster = {}
    for spk in clusters:
        sims = matrix[spk]
        ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
        top1_name, top1 = ranked[0]
        top2_name, top2 = ranked[1]
        others = np.array([v for _, v in ranked[1:]], dtype=np.float64)
        z = float((top1 - others.mean()) / (others.std() + 1e-9))
        per_cluster[spk] = {
            "top1_name": top1_name, "top1": round(top1, 4),
            "top2_name": top2_name, "top2": round(top2, 4),
            "margin": round(top1 - top2, 4),
            "z": round(z, 2),
            "n_segments": len(segs_by_spk[spk]),
            "duration_s": round(sum(e - s for s, e in segs_by_spk[spk]), 1),
            "ranked": [[n, round(v, 4)] for n, v in ranked],
        }

    # ---- Impostor floor ----------------------------------------------------
    # Every score except each cluster's own top1. These are overwhelmingly
    # wrong pairings, so their distribution is what "no signal" looks like.
    impostor = np.array(
        [v for spk in clusters for n, v in matrix[spk].items() if n != per_cluster[spk]["top1_name"]],
        dtype=np.float64,
    )
    floor = {
        "n": int(impostor.size),
        "mean": round(float(impostor.mean()), 4),
        "std": round(float(impostor.std()), 4),
        "p95": round(float(np.percentile(impostor, 95)), 4),
        "p99": round(float(np.percentile(impostor, 99)), 4),
        "max": round(float(impostor.max()), 4),
    }

    # ---- Report ------------------------------------------------------------
    print(f"\n{'cluster':20s} {'best match':12s} {'top1':>7s} {'top2':>7s} {'margin':>7s} {'z':>6s}  {'dur':>6s}")
    print("-" * 76)
    for spk in sorted(clusters, key=lambda s: per_cluster[s]["top1"], reverse=True):
        c = per_cluster[spk]
        flag = "  <-- clears p99" if c["top1"] > floor["p99"] else ""
        print(f"{spk:20s} {c['top1_name']:12s} {c['top1']:7.3f} {c['top2']:7.3f} "
              f"{c['margin']:7.3f} {c['z']:6.2f}  {c['duration_s']:5.1f}s{flag}")

    print(f"\nImpostor floor over {floor['n']} non-top pairings:")
    print(f"  mean {floor['mean']:.3f}   std {floor['std']:.3f}   "
          f"p95 {floor['p95']:.3f}   p99 {floor['p99']:.3f}   max {floor['max']:.3f}")

    z_thresh = cfg["evaluate"]["z_threshold"]
    clears = [s for s in clusters if per_cluster[s]["top1"] > floor["p99"] and per_cluster[s]["z"] >= z_thresh]
    students = [s for s in clears if per_cluster[s]["top1_name"] != "Prof"]
    print(f"\nClusters clearing p99 AND z>={z_thresh}: {len(clears)}  "
          f"(of which non-Prof: {len(students)})")
    if students:
        print("  " + ", ".join(f"{s}->{per_cluster[s]['top1_name']}" for s in students))

    # Which enrolled speakers were never anyone's best match
    matched_names = {per_cluster[s]["top1_name"] for s in clusters}
    unmatched = [n for n in names if n not in matched_names]
    print(f"\nEnrolled but never a top-1 match ({len(unmatched)}): {', '.join(unmatched) or 'none'}")

    out = {
        "phase": 0,
        "lecture": cfg["lecture"],
        "n_turns": len(turns),
        "enrolled": names,
        "clusters": clusters,
        "matrix": {s: {n: round(v, 4) for n, v in matrix[s].items()} for s in clusters},
        "per_cluster": per_cluster,
        "impostor_floor": floor,
        "clears_floor": clears,
        "clears_floor_students": students,
        "never_matched": unmatched,
    }
    with open(results / "baseline.json", "w") as f:
        json.dump(out, f, indent=2)
    np.savez(results / "cluster_embeddings.npz", **embeddings)
    print(f"\n  → wrote results/baseline.json")
    print(f"  → wrote results/cluster_embeddings.npz")


if __name__ == "__main__":
    main()
