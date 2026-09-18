"""Turn human-confirmed identifications into better voiceprints.

The insight this exists for: when a professor confirms "that cluster is Clive",
she has not only fixed one screen — she has handed the system 33 seconds of
Clive's voice *recorded through the room microphone*. That is precisely the data
the enrollment clips lack, because those were recorded on students' phones. Each
confirmed cluster shrinks the channel gap that made matching hard in the first
place.

THE SAFETY RULE, which is the whole design:

    Only audio a human explicitly vouched for may enrich a voiceprint.
    Never the system's own predictions.

Ingesting high-confidence predictions would be trivial to add and is the single
worst thing that could be done here. A wrong voiceprint wins more clusters, which
get ingested, which bias it further — and the rot is invisible because the
similarity scores keep rising. Every clip written by this script traces to a
`ground_truth.jsonl` record with a human's verdict on it.

Run:  python3 scripts/apply_learning.py [--lecture lecture_01.wav] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from candidates import cosine, rank_candidates  # noqa: E402

ENROLL_DIR = ROOT / "audio" / "enrollment"
RETIRED_DIR = ENROLL_DIR / "_retired"
LEDGER = ROOT / "data" / "enrollment_ledger.json"
GROUND_TRUTH = ROOT / "data" / "ground_truth.jsonl"
NAME_MAP = ROOT / "data" / "name_map.json"

INGESTIBLE = {"confirm", "correct", "unenrolled"}


def slug(name: str) -> str:
    """Display name -> filename-safe token with no underscores.

    enroll.py splits on the first "_", so the slug itself must not contain one.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower()) or "unnamed"


def load_ledger() -> dict:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else {}


def latest_verdicts() -> dict[str, dict]:
    """Replay the append-only log; last write per cluster wins."""
    out: dict[str, dict] = {}
    if not GROUND_TRUTH.exists():
        return out
    for line in GROUND_TRUTH.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("cluster_id"):
            out[rec["cluster_id"]] = rec
    return out


def spans_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lecture", default="lecture_01.wav")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = json.loads((ROOT / "config.json").read_text())
    lc = cfg.get("learning", {})
    min_sec = lc.get("min_speech_seconds", 8)
    min_sec_new = lc.get("min_speech_seconds_new_speaker", 15)
    max_clips = lc.get("max_learned_clips_per_person", 5)
    min_purity = lc.get("min_cluster_purity", 0.5)

    lecture_path = ROOT / "audio" / "lectures" / args.lecture
    if not lecture_path.exists():
        sys.exit(f"lecture audio not found: {lecture_path}")

    verdicts = latest_verdicts()
    ledger = load_ledger()
    known_vp = json.loads((ROOT / "data" / "voiceprints.json").read_text())

    if not verdicts:
        sys.exit("No human verdicts in data/ground_truth.jsonl — nothing to learn from.")

    # --- Retraction: a cluster whose verdict changed since it was ingested ----
    retired = 0
    for fname, entry in list(ledger.items()):
        cur = verdicts.get(entry["cluster_id"])
        if cur and cur["id"] != entry["source_record_id"]:
            src = ENROLL_DIR / fname
            if src.exists():
                RETIRED_DIR.mkdir(parents=True, exist_ok=True)
                if not args.dry_run:
                    shutil.move(str(src), str(RETIRED_DIR / fname))
            if not args.dry_run:
                del ledger[fname]
            retired += 1
            print(f"  RETIRED {fname} — verdict on {entry['cluster_id']} changed")

    # --- Select what to ingest ------------------------------------------------
    all_spans = [(c, tuple(s)) for c, r in verdicts.items() for s in (r.get("spans") or [])]
    ingested = {e["source_record_id"] for e in ledger.values()}
    per_person = {}
    for e in ledger.values():
        per_person[e["name"]] = per_person.get(e["name"], 0) + 1

    to_write, skipped = [], []
    for cid, rec in sorted(verdicts.items()):
        name = rec.get("corrected_name")
        why = None
        if rec.get("verdict") not in INGESTIBLE or not name:
            why = f"verdict={rec.get('verdict')}"
        elif rec["id"] in ingested:
            why = "already ingested"
        elif rec.get("verdict") == "unenrolled" and not rec.get("consent_to_enroll"):
            # Naming someone in a transcript and building a biometric voiceprint
            # of them are different acts. Absent explicit consent, do only the first.
            why = "unenrolled without consent_to_enroll — named only, no voiceprint"
        elif per_person.get(name, 0) >= max_clips:
            why = f"already has {max_clips} learned clips"
        if why:
            skipped.append((cid, name, why))
            continue

        spans = [tuple(s) for s in (rec.get("spans") or [])]
        # Crosstalk is the highest-yield contamination source: audio where two
        # people overlap would teach BOTH voiceprints the wrong thing.
        clean = [s for s in spans
                 if not any(spans_overlap(s, o) for c2, o in all_spans if c2 != cid)]
        dropped = len(spans) - len(clean)
        clean = [s for s in clean if s[1] - s[0] >= 0.5]
        total = sum(b - a for a, b in clean)
        floor = min_sec_new if name not in known_vp else min_sec
        if total < floor:
            skipped.append((cid, name, f"only {total:.1f}s usable (need {floor}s"
                                       f"{', ' + str(dropped) + ' spans dropped as overlapping' if dropped else ''})"))
            continue
        to_write.append((cid, rec, name, clean, total, dropped))

    if not to_write:
        print("\nNothing new to ingest.")
        for cid, name, why in skipped:
            print(f"  skip {cid} ({name}): {why}")
        return

    # --- Purity check + extraction -------------------------------------------
    from speechbrain.inference.speaker import EncoderClassifier
    from common import MODELS, load_wav_mono16k
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(MODELS / "ecapa"), run_opts={"device": "cpu"})
    sig, sr = load_wav_mono16k(str(lecture_path))

    import torch
    written = []
    for cid, rec, name, spans, total, dropped in to_write:
        # A diarization cluster holding two people would be laundered into one
        # voiceprint by whole-cluster relabelling. Catch it before writing.
        if len(spans) >= 2:
            embs = [enc.encode_batch(sig[:, int(a * sr):int(b * sr)]).squeeze().detach().cpu().numpy()
                    for a, b in spans if b - a >= 1.0]
            if len(embs) >= 2:
                pairs = [cosine(embs[i], embs[j])
                         for i in range(len(embs)) for j in range(i + 1, len(embs))]
                purity = float(np.mean(pairs))
                if purity < min_purity:
                    skipped.append((cid, name, f"cluster purity {purity:.2f} < {min_purity} "
                                               "— may contain two speakers"))
                    continue

        sl = slug(name)
        n = per_person.get(name, 0) + 1
        fname = f"{sl}_{Path(args.lecture).stem}c{n}.wav"
        dest = ENROLL_DIR / fname
        if not args.dry_run:
            parts = []
            tmp = ROOT / "data" / "_tmp_learn"
            tmp.mkdir(exist_ok=True)
            for i, (a, b) in enumerate(spans):
                p = tmp / f"{i}.wav"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(a), "-t", str(b - a),
                                "-i", str(lecture_path), "-ac", "1", "-ar", "16000",
                                "-c:a", "pcm_s16le", str(p)], check=True, stdin=subprocess.DEVNULL)
                parts.append(p)
            lst = tmp / "list.txt"
            lst.write_text("".join(f"file '{p.name}'\n" for p in parts))
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
                            "-i", str(lst), "-ac", "1", "-ar", "16000",
                            "-c:a", "pcm_s16le", str(dest)], check=True,
                           cwd=str(tmp), stdin=subprocess.DEVNULL)
            shutil.rmtree(tmp, ignore_errors=True)
            ledger[fname] = {
                "source_record_id": rec["id"], "lecture": args.lecture, "cluster_id": cid,
                "spans": spans, "name": name, "speech_seconds": round(total, 1),
                "audio_sha1": hashlib.sha1(dest.read_bytes()).hexdigest()[:16],
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        per_person[name] = per_person.get(name, 0) + 1
        written.append((fname, name, total, dropped))
        print(f"  + {fname}  ({name}, {total:.1f}s"
              f"{f', {dropped} overlapping spans dropped' if dropped else ''})")

    for cid, name, why in skipped:
        print(f"  skip {cid} ({name}): {why}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return
    if not written:
        print("\nNothing passed the quality gates.")
        return

    # --- Name map, backup, re-enroll -----------------------------------------
    nm = json.loads(NAME_MAP.read_text()) if NAME_MAP.exists() else {}
    for _, name, _, _ in written:
        nm[slug(name)] = name
    NAME_MAP.write_text(json.dumps(nm, indent=2, ensure_ascii=False))
    LEDGER.write_text(json.dumps(ledger, indent=2, ensure_ascii=False))

    backups = ROOT / "data" / "voiceprints_backups"
    backups.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    shutil.copy2(ROOT / "data" / "voiceprints.json", backups / f"{stamp}.json")

    before = {k: np.array(v, np.float32) for k, v in known_vp.items()}
    print("\nre-enrolling…")
    subprocess.run([sys.executable, str(ROOT / "pipeline" / "enroll.py")], check=True,
                   cwd=str(ROOT / "pipeline"))
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_confusables.py")], check=True)
    after = {k: np.array(v, np.float32)
             for k, v in json.loads((ROOT / "data" / "voiceprints.json").read_text()).items()}

    # --- Did it actually help? ------------------------------------------------
    # A cluster whose own audio went into the voiceprint will of course match it
    # better — that is memorisation, not learning, and quoting it as an
    # improvement would be self-congratulatory nonsense. Only clusters marked
    # HELD-OUT below are evidence that anything generalised.
    learned_from = {e["cluster_id"] for e in ledger.values()}

    print("\n" + "=" * 78)
    print("BEFORE / AFTER — every human-verified cluster, re-scored")
    print("=" * 78)
    print(f"{'cluster':<9}{'truth':<22}{'before':>8}{'after':>8}{'rank':>6}  evidence")
    print("-" * 78)
    held_out_deltas = []
    for cid, rec in sorted(verdicts.items()):
        truth = rec.get("corrected_name")
        spans = [tuple(s) for s in (rec.get("spans") or [])]
        if not truth or not spans:
            continue
        ch = [sig[:, int(a * sr):int(b * sr)] for a, b in spans if b > a]
        if not ch:
            continue
        emb = enc.encode_batch(torch.cat(ch, dim=1)).squeeze().detach().cpu().numpy()
        b_sim = cosine(emb, before[truth]) if truth in before else float("nan")
        a_rank = rank_candidates(emb, after, top_n=len(after))
        a_sim = next((c["sim"] for c in a_rank if c["name"] == truth), float("nan"))
        pos = next((i + 1 for i, c in enumerate(a_rank) if c["name"] == truth), None)
        if cid in learned_from:
            tag = "trained on (circular)"
        elif np.isnan(b_sim):
            tag = "not enrolled"
        else:
            tag = "HELD-OUT"
            held_out_deltas.append(a_sim - b_sim)
        arrow = "" if np.isnan(b_sim) else (" ↑" if a_sim > b_sim else " ↓" if a_sim < b_sim else " =")
        print(f"{cid:<9}{truth:<22}{b_sim:>8.3f}{a_sim:>8.3f}{str(pos):>6}{arrow}  {tag}")

    print("-" * 78)
    if held_out_deltas:
        mean_d = float(np.mean(held_out_deltas))
        print(f"HELD-OUT clusters (audio NOT in any voiceprint): n={len(held_out_deltas)}, "
              f"mean change {mean_d:+.3f}")
        print("This is the only line that shows generalisation. Everything marked")
        print("'trained on' is the model recognising its own training audio.")
    else:
        print("No held-out clusters — every verified cluster contributed audio, so")
        print("NOTHING here demonstrates generalisation. Verify on a fresh window.")

    print(f"\n{len(written)} clip(s) learned, {retired} retired. "
          f"Backup: data/voiceprints_backups/{stamp}.json")


if __name__ == "__main__":
    main()
