"""Score the verified experiment windows and publish them to the dashboard.

Takes data/experiments.json (human-verified speaker identities), classifies each
student turn for participation quality, applies the scoring + anti-gaming rules
from config.json, and writes the three files the dashboard reads:

    data/named_turns.json    (absolute lecture times — professor timeline)
    data/scored_turns.json   (quality-labelled student turns)
    data/scores.json         (per-student + per-team totals)

The previous contents are the 2026-08-16 pilot, whose student turns were 100%
Unknown. They are moved to data/_pilot_backup/ rather than deleted.

Student-Z is included: she has no voiceprint, so the system could not name her, but
her identity is confirmed and she did participate. Excluding her would under-count
a real contributor — she is flagged `unenrolled` instead.

Run:  python3 scripts/score_experiments.py
"""
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "pipeline"))

from classify import LABELS, SYSTEM, MODEL  # noqa: E402


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not found in .env")

    cfg = json.loads((ROOT / "config.json").read_text())
    weights, ag = cfg["quality_weights"], cfg["anti_gaming"]
    prof_label = cfg["professor_label"]
    data = json.loads((ROOT / "data" / "experiments.json").read_text())

    # Preserve the pilot rather than overwrite it.
    backup = ROOT / "data" / "_pilot_backup"
    backup.mkdir(exist_ok=True)
    for f in ("named_turns.json", "scored_turns.json", "scores.json", "source_audio.json"):
        src = ROOT / "data" / f
        if src.exists() and not (backup / f).exists():
            shutil.copy2(src, backup / f)
    print(f"pilot backed up -> data/_pilot_backup/")

    # Flatten both windows onto the absolute lecture clock.
    named = []
    for w in data["windows"]:
        off = {"w1": 3240.0, "w2": 3930.0}[w["id"]]
        # Cluster ids must NOT contain the system's guess: the moment a professor
        # relabels a cluster the id would stop matching its own name, and re-running
        # would silently renumber everything. Assign stable ordinals by order of
        # first appearance instead.
        ordinals: dict[str, int] = {}
        key = lambda t: t.get("clusterLabel", t["systemGuess"])  # noqa: E731
        for t in sorted(w["turns"], key=lambda x: x["start"]):
            ordinals.setdefault(key(t), len(ordinals))
        for t in w["turns"]:
            cid = f"{w['id']}_c{ordinals[key(t)]}"
            named.append({
                "name": t["name"],
                "speaker_id": cid,
                "cluster_id": cid,
                "start": round(t["start"] + off, 2),
                "end": round(t["end"] + off, 2),
                "text": t["text"],
                "confidence": t["sim"],
                "candidates": t.get("candidates") or [
                    {"name": t["systemGuess"], "sim": t["sim"], "z": None}
                ],
                "status": t["status"],
                "window": w["label"],
            })
    named.sort(key=lambda t: t["start"])
    (ROOT / "data" / "named_turns.json").write_text(json.dumps(named, indent=2, ensure_ascii=False))
    print(f"wrote named_turns.json ({len(named)} turns)")

    # clusters.json is what tells the dashboard which turns need a human. The
    # normal pipeline gets it from match.py; the experiment path has to build the
    # equivalent here or no review flags would ever fire.
    from candidates import decide, load_policy, voiceprints_hash
    policy = load_policy(cfg)
    clusters = {}
    for t in named:
        cid = t["cluster_id"]
        c = clusters.setdefault(cid, {
            "cluster_id": cid, "spans": [], "candidates": t["candidates"],
            "decision": t["name"], "decision_sim": t["confidence"],
            "decision_z": (t["candidates"][0] or {}).get("z"),
            "match_status": decide(t["candidates"], policy),
            # Already human-verified, so never spot-checked again.
            "audit_sampled": False,
            "human_verified": t["status"],
        })
        c["spans"].append([t["start"], t["end"]])
    for c in clusters.values():
        c["spans"].sort()
        c["n_turns"] = len(c["spans"])
        c["speech_seconds"] = round(sum(b - a for a, b in c["spans"]), 1)

    (ROOT / "data" / "clusters.json").write_text(json.dumps({
        "lecture": "lecture_01.wav",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "voiceprints_hash": voiceprints_hash(json.loads((ROOT / "data" / "voiceprints.json").read_text())),
        "policy": policy,
        "clusters": clusters,
    }, indent=2, ensure_ascii=False))
    for cid, c in sorted(clusters.items()):
        print(f"  {cid:<8} {c['decision']:<24} sim={c['decision_sim']:.3f}  {c['match_status']}")

    from openai import OpenAI
    client = OpenAI()

    scored = []
    for t in named:
        if t["name"] == prof_label:
            continue
        label = classify(client, t["text"])
        scored.append({
            "name": t["name"], "start": t["start"], "end": t["end"], "text": t["text"],
            "quality": label, "points": weights.get(label, 0), "status": t["status"],
        })
        print(f"  {t['name']:<24} {label:<16} +{weights.get(label,0)}  {t['text'][:46]}")
    (ROOT / "data" / "scored_turns.json").write_text(json.dumps(scored, indent=2, ensure_ascii=False))

    # Same anti-gaming rules as pipeline/score.py.
    scored.sort(key=lambda t: (t["name"], t["start"]))
    windows_seen, last_quality = defaultdict(list), {}
    students = defaultdict(lambda: {"total": 0.0, "raw_total": 0,
                                    "by_quality": defaultdict(int), "turns": 0})
    status_of = {}
    for t in scored:
        n = t["name"]
        status_of[n] = t["status"]
        pts = float(t["points"])
        recent = [s for s in windows_seen[n] if t["start"] - s <= ag["window_seconds"]]
        recent.append(t["start"])
        windows_seen[n] = recent
        if len(recent) > ag["max_turns_per_window"]:
            pts *= ag["overflow_multiplier"]
        if last_quality.get(n) == t["quality"]:
            pts *= ag["consecutive_same_quality_discount"]
        last_quality[n] = t["quality"]
        s = students[n]
        s["total"] += pts
        s["raw_total"] += t["points"]
        s["by_quality"][t["quality"]] += 1
        s["turns"] += 1

    out_students = sorted(
        ({"name": n, "total": round(v["total"], 1), "raw_total": v["raw_total"],
          "by_quality": dict(v["by_quality"]), "turns": v["turns"],
          "status": status_of.get(n, "confirmed")} for n, v in students.items()),
        key=lambda s: -s["total"],
    )
    teams_cfg = cfg.get("teams", {})
    out_teams = sorted(
        ({"team": tn, "members": mem,
          "total": round(sum(s["total"] for s in out_students if s["name"] in mem), 1)}
         for tn, mem in teams_cfg.items()),
        key=lambda t: -t["total"],
    )
    (ROOT / "data" / "scores.json").write_text(
        json.dumps({"students": out_students, "teams": out_teams}, indent=2, ensure_ascii=False))
    build_combined_audio(data, named, scored)

    print(f"\nwrote scores.json — {len(out_students)} students, {len(out_teams)} teams")
    for s in out_students:
        flag = "  (unenrolled)" if s["status"] == "unenrolled" else ""
        print(f"  {s['name']:<24} {s['total']:>5} pts  ({s['turns']} turns){flag}")
    if not out_teams:
        print("\nNOTE: config.json has no teams — the team leaderboard will be empty.")


def build_combined_audio(data: dict, named: list, scored: list) -> None:
    """Stitch the analysed windows into one small clip for the dashboard.

    The analysed windows total ~4.5 minutes, but named_turns.json carries
    absolute lecture times — so the dashboard was loading the whole 164 MB
    lecture just to play them. Concatenating the `_seg_*.wav` clips that
    whospoke.py already produced gives an ~8 MB file that loads instantly.

    Turn times are rewritten onto the combined clip's clock for playback sync,
    and the original lecture time is preserved as `abs_label` for display, so
    the professor still sees "0:54:06" rather than "0:00:06".
    """
    import subprocess

    lectures = ROOT / "audio" / "lectures"
    offsets = {"w1": 3240.0, "w2": 3930.0}
    parts, remap, cursor = [], {}, 0.0
    for w in data["windows"]:
        off = offsets[w["id"]]
        seg = lectures / f"_seg_{int(off)}_{int(off + round(w['duration']))}.wav"
        if not seg.exists():
            # Fall back to the full lecture rather than producing broken audio.
            matches = sorted(lectures.glob(f"_seg_{int(off)}_*.wav"))
            if not matches:
                print(f"  no _seg clip for {w['id']} — keeping full lecture as audio source")
                (ROOT / "data" / "source_audio.json").write_text(
                    json.dumps({"filename": "lecture_01.wav"}))
                return
            seg = matches[0]
        dur = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(seg)],
            capture_output=True, text=True, check=True).stdout.strip())
        parts.append(seg)
        remap[w["id"]] = (off, cursor)   # absolute window start -> combined start
        cursor += dur

    out = lectures / "_seg_combined.wav"
    tmp = ROOT / "data" / "_tmp_concat"
    tmp.mkdir(exist_ok=True)
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(lst), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                    str(out)], check=True, stdin=subprocess.DEVNULL)
    shutil.rmtree(tmp, ignore_errors=True)

    def fmt_abs(s: float) -> str:
        return f"{int(s)//3600}:{(int(s)%3600)//60:02d}:{int(s)%60:02d}"

    # Build the shift table from named_turns (which knows each turn's window),
    # then apply the SAME shift to scored_turns. getTimeline() joins the two on
    # `name@start`, so remapping one without the other would silently strip the
    # quality label off every turn.
    shift_for = {}
    for t in named:
        wid = t["cluster_id"].split("_")[0]
        abs_off, comb_off = remap[wid]
        shift_for[round(t["start"], 2)] = comb_off - abs_off
        t["abs_label"] = fmt_abs(t["start"])
        t["abs_start"] = t["start"]
        t["start"] = round(t["start"] - abs_off + comb_off, 2)
        t["end"] = round(t["end"] - abs_off + comb_off, 2)
    named.sort(key=lambda t: t["start"])
    (ROOT / "data" / "named_turns.json").write_text(json.dumps(named, indent=2, ensure_ascii=False))

    for s in scored:
        shift = shift_for.get(round(s["start"], 2))
        if shift is None:
            continue
        s["abs_start"] = s["start"]
        s["start"] = round(s["start"] + shift, 2)
        s["end"] = round(s["end"] + shift, 2)
    (ROOT / "data" / "scored_turns.json").write_text(json.dumps(scored, indent=2, ensure_ascii=False))
    (ROOT / "data" / "source_audio.json").write_text(json.dumps({"filename": out.name}))
    size_mb = out.stat().st_size / 1048576
    print(f"\nwrote {out.name} ({size_mb:.1f} MB, {cursor:.0f}s) from {len(parts)} _seg clips")
    print(f"  timeline remapped onto it; absolute lecture times kept as abs_label")


def classify(client, text: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    try:
        label = json.loads(resp.choices[0].message.content)["label"]
    except Exception:
        return "acknowledgement"
    return label if label in LABELS else "acknowledgement"


if __name__ == "__main__":
    main()
