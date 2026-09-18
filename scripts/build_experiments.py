"""Build data/experiments.json — the verified two-window experiment set.

Merges scripts/_whospoke_<start>_<end>.json (system output) with the ground truth
the user confirmed on 2026-08-30/31, so the dashboard shows what was actually
established rather than what the old (miscalibrated) threshold accepted.

Three statuses per turn:
  confirmed  — system named them AND the user verified it
  unenrolled — user identified the speaker; they have no voiceprint, so the
               system correctly declined to name them
  unverified — system output with no human check (none in this set)

Run:  python3 scripts/build_experiments.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Ground truth. Keyed by the raw best-guess name the system produced, per window,
# because that is the only stable join key back to the saved whospoke output.
WINDOWS = [
    {
        "id": "w1",
        "file": "_whospoke_3240_3360.json",
        "audio": "_seg_3240_3360.wav",
        "offset": 3240.0,
        "label": "Window 1 — 54:00 to 56:00",
        "note": "Discussion on system architecture and security. All three speakers "
                "scored above the old threshold.",
        "truth": {
            "Prof": ("Prof", "confirmed"),
            "Clive": ("Clive", "confirmed"),
            "Jayden": ("Jayden", "confirmed"),
        },
    },
    {
        "id": "w2",
        "file": "_whospoke_3930_4080.json",
        "audio": "_seg_3930_4080.wav",
        "offset": 3930.0,
        "label": "Window 2 — 1:05:30 to 1:08:00",
        "note": "Discussion on AI inference transparency and human complements. Two "
                "speakers here were REJECTED by the old 0.536 threshold and later "
                "confirmed correct — this window is why the threshold was revised.",
        "truth": {
            "Prof": ("Prof", "confirmed"),
            "Saieshwar": ("Saieshwar", "confirmed"),
            # Scored 0.340 / 0.225 — below the old floor, rejected, later confirmed.
            "Kaiting": ("Kaiting", "confirmed"),
            "Rija": ("Rija", "confirmed"),
            # Scored 0.143 and was correctly declined: the true speaker has no
            # voiceprint, so no threshold could have produced her name.
            "Richard": ("Student-Z", "unenrolled"),
        },
    },
]


def fmt(s: float) -> str:
    return f"{int(s)//3600}:{(int(s)%3600)//60:02d}:{int(s)%60:02d}"


def main() -> None:
    out = []
    for w in WINDOWS:
        raw = json.loads((ROOT / "scripts" / w["file"]).read_text())
        turns, speakers = [], {}
        for r in raw:
            best = r["raw_best"]
            name, status = w["truth"].get(best, (best, "unverified"))
            rel_start = r["start"] - w["offset"]
            rel_end = r["end"] - w["offset"]
            turns.append({
                "start": round(rel_start, 3),      # relative — drives audio sync
                "end": round(rel_end, 3),
                "absLabel": fmt(r["start"]),       # absolute — for display
                "name": name,
                "isProf": name == "Prof",
                "sim": round(r["sim"], 3),
                "status": status,
                "systemGuess": best,
                # Diarization cluster label, not the guessed name — a relabel must
                # not change the id. Older whospoke output lacks it; fall back to
                # the guess so those records still group correctly.
                "clusterLabel": r.get("cluster_id", best),
                "candidates": r.get("candidates", [{"name": best, "sim": round(r["sim"], 3), "z": None}]),
                "text": r["text"],
            })
            s = speakers.setdefault(name, {
                "name": name, "status": status, "sim": round(r["sim"], 3),
                "turns": 0, "seconds": 0.0, "systemGuess": best,
            })
            s["turns"] += 1
            s["seconds"] += rel_end - rel_start

        for s in speakers.values():
            s["seconds"] = round(s["seconds"], 1)

        turns.sort(key=lambda t: t["start"])
        out.append({
            "id": w["id"], "label": w["label"], "note": w["note"],
            "audio": w["audio"],
            "duration": round(max(t["end"] for t in turns), 1),
            "turns": turns,
            "speakers": sorted(speakers.values(), key=lambda s: -s["seconds"]),
        })

    dest = ROOT / "data" / "experiments.json"
    dest.write_text(json.dumps({"windows": out}, indent=2, ensure_ascii=False))
    print(f"wrote {dest.relative_to(ROOT)}")
    for w in out:
        names = ", ".join(f"{s['name']}({s['status'][:4]})" for s in w["speakers"])
        print(f"  {w['label']}: {len(w['turns'])} turns — {names}")


if __name__ == "__main__":
    main()
