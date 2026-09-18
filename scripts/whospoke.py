"""Identify who spoke in a time window of a lecture recording.

Standalone: reads data/voiceprints.json but writes nothing into data/, so it
never clobbers pipeline output the dashboard is serving.

Usage:  python3 scripts/whospoke.py 1:05:30 1:08:00 [audio/lectures/lecture_01.wav]
"""
import itertools, json, re, subprocess, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from candidates import cosine, decide, load_policy, rank_candidates  # noqa: E402
from common import load_wav_mono16k, MODELS  # noqa: E402
from diarize import diarize  # noqa: E402


def parse_ts(t: str) -> float:
    parts = [float(x) for x in t.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def fmt(s: float) -> str:
    return f"{int(s)//3600}:{(int(s)%3600)//60:02d}:{int(s)%60:02d}"


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    start, end = parse_ts(sys.argv[1]), parse_ts(sys.argv[2])
    src = sys.argv[3] if len(sys.argv) > 3 else str(ROOT / "audio/lectures/lecture_01.wav")
    dur = end - start
    seg_wav = ROOT / f"audio/lectures/_seg_{int(start)}_{int(end)}.wav"

    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(dur),
                    "-i", src, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                    str(seg_wav)], check=True, stdin=subprocess.DEVNULL)

    sig, sr = load_wav_mono16k(str(seg_wav))
    wav = sig.squeeze(0).numpy()
    print(f"window {fmt(start)}–{fmt(end)}  ({dur:.0f}s)  "
          f"RMS={np.sqrt(np.mean(wav**2)):.4f}\n")

    segs = diarize(str(seg_wav), chunk_seconds=0, batch_size=8)
    by = defaultdict(list)
    for a, b, lab in segs:
        by[lab].append((a, b))

    vp = {k: np.array(v, np.float32)
          for k, v in json.load(open(ROOT / "data/voiceprints.json")).items()}
    cfg = json.loads((ROOT / "config.json").read_text())
    policy = load_policy(cfg)
    names = [n for n in vp if n != cfg.get("professor_label", "Prof")]
    imp = np.array([cosine(vp[a], vp[b]) for a, b in itertools.combinations(names, 2)])
    IMAX, IP99 = imp.max(), np.percentile(imp, 99)

    from speechbrain.inference.speaker import EncoderClassifier
    enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                         savedir=str(MODELS / "ecapa"),
                                         run_opts={"device": "cpu"})
    from funasr import AutoModel
    sense = MODELS / "SenseVoiceSmall"
    asr = AutoModel(model=str(sense) if (sense / "model.pt").exists() else "iic/SenseVoiceSmall",
                    hub="ms", device="cpu", disable_update=True)
    clean = lambda t: re.sub(r"<\|[^|]*\|>", "", t or "").strip()

    turn_rows = []
    rows = []
    for lab, ss in by.items():
        ch = [sig[:, int(a * sr):int(b * sr)] for a, b in ss if b > a]
        spoken = sum(c.shape[1] for c in ch) / sr
        e = enc.encode_batch(torch.cat(ch, dim=1)).squeeze().detach().cpu().numpy()
        ranked = rank_candidates(e, vp, top_n=policy["top_n_candidates"])
        rows.append((lab, spoken, ranked, ss))

    print("=" * 78)
    print(f"WHO SPOKE   {fmt(start)}–{fmt(end)}      impostor floor: p99={IP99:.3f} max={IMAX:.3f}")
    print("=" * 78)
    for lab, spoken, ranked, ss in sorted(rows, key=lambda r: -r[1]):
        top, second = ranked[0], ranked[1]
        # Verdict comes from the shared review policy so this CLI and the
        # dashboard can never disagree about what "confident" means.
        status = decide(ranked, policy)
        if status == "accepted":
            verdict = "CONFIDENT"
        elif status == "review":
            verdict = f"REVIEW (next: {second['name']} {second['sim']:.3f})"
        else:
            verdict = "UNKNOWN (below review floor — may be unenrolled)"
        z = top.get("z")
        zs = "n/a" if z is None else f"{z:.2f}"
        print(f"\n{lab}  {spoken:.0f}s speech, {len(ss)} turns   -> "
              f"{top['name']} {top['sim']:.3f} (z={zs})   [{verdict}]")
        for c in ranked[:4]:
            cz = "n/a" if c.get("z") is None else f"{c['z']:.2f}"
            print(f"      {c['name']:<12} {c['sim']:6.3f}  z={cz}")
        for a, b in ss:
            c_ = wav[int(a * sr):int(b * sr)]
            if len(c_) < int(0.4 * sr):
                continue
            o = asr.generate(input=c_, fs=sr, language="en", use_itn=True)
            t = clean(o[0].get("text", "")) if o else ""
            if t:
                print(f"        [{fmt(a + start)}] {t[:92]}")
                turn_rows.append((a + start, b + start, top["name"], top["sim"],
                                  verdict, t, lab, ranked[:3]))

    # Chronological view: who said what, in order, with identity resolved.
    turn_rows.sort(key=lambda r: r[0])
    print("\n" + "=" * 78)
    print("CHRONOLOGICAL  —  speaker by time frame")
    print("=" * 78)
    for a, b, name, sim, verdict, text, lab, cands in turn_rows:
        trusted = verdict.startswith("CONFIDENT")
        who = name if trusted else f"{name} (best guess)"
        # Confirmed 2026-08-31: a sub-floor score can mean the true speaker has no
        # voiceprint at all (Student-Z, unenrolled, was "best guessed"
        # as Richard at 0.14). Never present these as identifications.
        tag = "" if trusted else f"   [{sim:.2f} — {name} unconfirmed; may be UNENROLLED]"
        print(f"{fmt(a)}–{fmt(b)}  {who:<24}{tag}")
        print(f"                       {text[:88]}")
    out = ROOT / f"scripts/_whospoke_{int(start)}_{int(end)}.json"
    json.dump([{"start": a, "end": b,
                "name": (n if v.startswith("CONFIDENT") else f"{n} (best guess)"),
                "raw_best": n, "sim": s_, "verdict": v, "text": t,
                "cluster_id": lab, "candidates": cands}
               for a, b, n, s_, v, t, lab, cands in turn_rows], open(out, "w"), indent=2)
    print(f"\nsaved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
