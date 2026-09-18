"""Stage 3 — Speaker match.

Maps anonymous diarization labels (Speaker_0...) to enrolled names.

Strategy: aggregate ALL audio for each diarized speaker_id, embed once with
ECAPA, and match that aggregate against the voiceprint DB. Matching per-speaker
(not per-turn) is far more robust to short "I agree" utterances — a 1.5s turn
inherits the identity of its whole speaker cluster.

    max cosine sim >= threshold  -> assign best-matching name
    else                         -> "Unknown"  (honest fallback)

Also reports the % of turns left Unknown — the prototype's key honesty metric
for how well speaker-ID is holding up (the go/no-go signal for scaling).

Every cluster's full ranked candidate list is kept, not just the winner — the
dashboard's correction picker offers the runner-ups, and a stored correction
records where the true name actually ranked.

Output: data/named_turns.json
    [ { "name": "Priya", "start": .., "end": .., "text": "..", "confidence": ..,
        "cluster_id": "SPEAKER_00", "candidates": [{name, sim, z}, ...] }, ... ]
        data/clusters.json   (per-cluster decisions + spans, for the picker)

Run:  python3 pipeline/match.py [audio/lectures/lecture_01.wav]
"""
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from candidates import (
    audit_sampled, cosine, decide, load_policy, rank_candidates, voiceprints_hash,
)
from common import (
    AUDIO_LECTURES, TURNS, VOICEPRINTS, NAMED_TURNS, CLUSTERS, MODELS,
    load_config, read_json, write_json, pick_device_speechbrain,
)


def main() -> None:
    cfg = load_config()
    threshold = cfg["match_threshold"]
    policy = load_policy(cfg)

    audio = sys.argv[1] if len(sys.argv) > 1 else None
    if audio is None:
        candidates = sorted(AUDIO_LECTURES.glob("*.wav"))
        if not candidates:
            sys.exit(f"No lecture audio found in {AUDIO_LECTURES}. Pass the same file used in transcribe.py.")
        audio = str(candidates[0])

    turns = read_json(TURNS)
    voiceprints = {k: np.array(v, dtype=np.float32) for k, v in read_json(VOICEPRINTS).items()}

    device = pick_device_speechbrain()
    print(f"Match — device={device}  threshold={threshold}  speakers_enrolled={len(voiceprints)}")

    from speechbrain.inference.speaker import EncoderClassifier
    from common import load_wav_mono16k

    signal, sr = load_wav_mono16k(audio)  # [1, N] mono 16k via soundfile

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(MODELS / "ecapa"),
        run_opts={"device": device},
    )

    # Group turn-segments by diarized speaker_id, embed each segment, average.
    segs_by_spk = defaultdict(list)
    for t in turns:
        segs_by_spk[t["speaker_id"]].append((t["start"], t["end"]))

    spk_identity = {}
    spk_candidates = {}
    clusters = {}
    for spk, segs in segs_by_spk.items():
        emb = embed_segments(encoder, signal, sr, segs)
        ranked = rank_candidates(emb, voiceprints, top_n=policy["top_n_candidates"])
        spk_candidates[spk] = ranked
        top = ranked[0] if ranked else {"name": None, "sim": 0.0, "z": None}
        status = decide(ranked, policy)

        # The legacy `match_threshold` still decides the NAME so existing
        # downstream stages behave identically; `status` is the new, separate
        # signal that drives review. Keeping them independent means adding the
        # picker cannot change anyone's score.
        if top["name"] is not None and top["sim"] >= threshold:
            spk_identity[spk] = (top["name"], top["sim"])
        else:
            spk_identity[spk] = ("Unknown", top["sim"])

        clusters[spk] = {
            "cluster_id": spk,
            "spans": [[round(a, 2), round(b, 2)] for a, b in sorted(segs)],
            "n_turns": len(segs),
            "speech_seconds": round(sum(b - a for a, b in segs), 1),
            "candidates": ranked,
            "decision": spk_identity[spk][0],
            "decision_sim": top["sim"],
            "decision_z": top.get("z"),
            "match_status": status,
            "audit_sampled": audit_sampled(spk, policy["audit_rate"]),
        }
        z = top.get("z")
        print(f"  {spk}: -> {spk_identity[spk][0]}  (sim={top['sim']:.3f} "
              f"z={z if z is None else f'{z:.2f}'}  {status})")

    named = []
    for t in turns:
        name, conf = spk_identity[t["speaker_id"]]
        named.append(
            {
                "name": name,
                "speaker_id": t["speaker_id"],
                "cluster_id": t["speaker_id"],
                "start": t["start"],
                "end": t["end"],
                "text": t["text"],
                "confidence": round(conf, 3),
                # Denormalised so the dashboard's timeline endpoint needs one read.
                "candidates": spk_candidates.get(t["speaker_id"], [])[:3],
            }
        )

    # Context attribution pass: if a professor turn mentions a student's name,
    # the immediately following Unknown turn is likely that student.
    enrolled_names = list(voiceprints.keys())
    prof = cfg["professor_label"]
    for i, t in enumerate(named):
        if t["name"] != "Unknown":
            continue
        # Look at the turn immediately before this one.
        prev = named[i - 1] if i > 0 else None
        if prev and prev["name"] in (prof, "Unknown"):
            mentioned = _find_name_mention(prev["text"], enrolled_names)
            if mentioned:
                t["name"] = mentioned
                t["attribution"] = "context"
                print(f"  {t['speaker_id']}: context-attributed -> {mentioned}  ('{prev['text'][:45]}...')")

    write_json(NAMED_TURNS, named)
    write_json(CLUSTERS, {
        "lecture": Path(audio).name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "voiceprints_hash": voiceprints_hash(read_json(VOICEPRINTS)),
        "policy": policy,
        "clusters": clusters,
    })

    unknown = sum(1 for t in named if t["name"] == "Unknown")
    pct = 100.0 * unknown / len(named) if named else 0.0
    print(f"Matched {len(named)} turns. Unknown: {unknown}/{len(named)} ({pct:.1f}%)  <-- honesty metric")

    needs_review = [c for c in clusters.values()
                    if c["match_status"] != "accepted" or c["audit_sampled"]]
    print(f"{len(needs_review)}/{len(clusters)} clusters flagged for human review.")


def _find_name_mention(text: str, names: list) -> str | None:
    """Return the first enrolled name found in text (case-insensitive), else None."""
    lower = text.lower()
    for name in names:
        if name.lower() in lower:
            return name
    return None


def embed_segments(encoder, signal, sr, segs) -> np.ndarray:
    """Concatenate all of a speaker's audio segments, embed as one."""
    import torch

    chunks = []
    for start, end in segs:
        a, b = int(start * sr), int(end * sr)
        if b > a:
            chunks.append(signal[:, a:b])
    if not chunks:
        return np.zeros(192, dtype=np.float32)
    cat = torch.cat(chunks, dim=1)
    emb = encoder.encode_batch(cat).squeeze().detach().cpu().numpy()
    return emb


if __name__ == "__main__":
    main()
