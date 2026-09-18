"""Find pairs of enrolled speakers whose voiceprints are dangerously similar.

When two students' reference vectors sit close together, a correct top-1 match
drags its confusable partner up with it — the runner-up's score is leakage, not
an independent competing hypothesis. Confirmed case: Jayden and Justin score
0.421 against each other, which is why a cluster that was genuinely Jayden
(0.541) showed Justin at 0.390 behind it.

Clusters whose top-2 are a known pair get sent for human review even when the
score looks healthy, because that is exactly the confident-but-ambiguous case a
similarity threshold cannot catch.

Reads data/voiceprints.json only — no audio, well under a second.

Run:  python3 scripts/build_confusables.py
"""
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from candidates import cosine, voiceprints_hash  # noqa: E402


def main() -> None:
    cfg = json.loads((ROOT / "config.json").read_text())
    conf_cfg = cfg.get("confusables", {})
    min_sim = conf_cfg.get("min_sim", 0.35)
    pct = conf_cfg.get("percentile", 95)
    prof = cfg.get("professor_label", "Prof")

    raw = json.loads((ROOT / "data" / "voiceprints.json").read_text())
    vp = {k: np.array(v, dtype=np.float32) for k, v in raw.items()}
    names = [n for n in vp if n != prof]
    if len(names) < 3:
        sys.exit(f"Only {len(names)} students enrolled — nothing to compare.")

    pairs = [(cosine(vp[a], vp[b]), a, b) for a, b in itertools.combinations(names, 2)]
    sims = np.array([p[0] for p in pairs])
    cutoff = float(np.percentile(sims, pct))

    # Both gates matter. The percentile alone flags the top 5% of ANY cohort —
    # in a small class that means flagging pairs that aren't actually similar.
    # The absolute floor alone would flag nothing in a well-separated cohort.
    threshold = max(cutoff, min_sim)
    flagged = sorted((p for p in pairs if p[0] >= threshold), reverse=True)

    by_name: dict[str, list] = {}
    for s, a, b in flagged:
        by_name.setdefault(a, []).append({"name": b, "sim": round(s, 3)})
        by_name.setdefault(b, []).append({"name": a, "sim": round(s, 3)})

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "voiceprints_hash": voiceprints_hash(raw),
        "n_speakers": len(names),
        "threshold_used": round(threshold, 3),
        "stats": {
            "n_pairs": len(pairs),
            "mean": round(float(sims.mean()), 3),
            "p50": round(float(np.percentile(sims, 50)), 3),
            "p95": round(float(np.percentile(sims, 95)), 3),
            "p99": round(float(np.percentile(sims, 99)), 3),
            "max": round(float(sims.max()), 3),
        },
        "pairs": [{"a": a, "b": b, "sim": round(s, 3)} for s, a, b in flagged],
        "by_name": by_name,
    }
    dest = ROOT / "data" / "confusables.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"wrote {dest.relative_to(ROOT)}")
    print(f"  {len(names)} students, {len(pairs)} pairs")
    print(f"  impostor: mean={out['stats']['mean']} p95={out['stats']['p95']} max={out['stats']['max']}")
    print(f"  threshold used: {threshold:.3f}  (p{pct}={cutoff:.3f}, floor={min_sim})")
    print(f"  {len(flagged)} confusable pair(s):")
    for s, a, b in flagged[:12]:
        print(f"    {a:<12} <-> {b:<12} {s:.3f}")


if __name__ == "__main__":
    main()
