"""Stage 2 — Transcribe + diarize.

Speaker-change detection is handled by pyannote.audio 3.1 (diarize.py) — it
detects turns from the voice itself, not from silence, so it correctly splits
speakers even when they talk with little or no gap between them. SenseVoiceSmall
(FunASR) then transcribes each diarized segment. speaker_id is pyannote's own
per-file cluster label (e.g. "SPEAKER_00"), reused across all of that
speaker's segments in the file.

Output: data/turns.json
    [ { "speaker_id": "SPEAKER_00", "start": 0.4, "end": 3.8, "text": "..." }, ... ]

Run:  python3 pipeline/transcribe.py [audio/lectures/lecture_01.wav]
"""
import sys
from pathlib import Path

from common import (
    AUDIO_LECTURES, TURNS, SOURCE_AUDIO, MODELS, load_config, write_json, pick_device,
)
from diarize import diarize


def main() -> None:
    audio = sys.argv[1] if len(sys.argv) > 1 else None
    if audio is None:
        candidates = sorted(AUDIO_LECTURES.glob("*.wav"))
        if not candidates:
            sys.exit(f"No lecture audio found in {AUDIO_LECTURES}. Pass a path or add a .wav.")
        audio = str(candidates[0])

    device = pick_device()
    print(f"Transcribe — device={device}  file={audio}")

    # Record which file this run's turns/scores belong to, so the dashboard
    # can serve it back to the professor for playback.
    write_json(SOURCE_AUDIO, {"filename": Path(audio).name})

    import numpy as np
    from funasr import AutoModel
    from common import load_wav_mono16k

    # Two-step pipeline:
    #   1. pyannote.audio detects real speaker-change points (not just silence).
    #   2. SenseVoiceSmall transcribes each diarized segment separately.
    # match.py (SpeechBrain ECAPA) resolves the real speaker NAME per
    # pyannote-assigned speaker_id downstream.
    sense = MODELS / "SenseVoiceSmall"
    sense_model = str(sense) if (sense / "model.pt").exists() else "iic/SenseVoiceSmall"

    asr = AutoModel(model=sense_model, hub="ms", device=device, disable_update=True)

    # Load audio once (soundfile) for slicing.
    sig, sr = load_wav_mono16k(audio)        # torch [1, N]
    wav = sig.squeeze(0).cpu().numpy()       # np [N]

    # Real speaker-change detection (not silence-based) — see diarize.py.
    # chunk_seconds caps diarization's peak memory; 0 restores the single-pass
    # path (better clustering, but needs RAM proportional to the whole file).
    cfg = load_config()
    segments = diarize(
        audio,
        chunk_seconds=cfg.get("diarize_chunk_seconds", 0),
        batch_size=cfg.get("diarize_batch_size", 8),
    )  # [(start_s, end_s, speaker_label), ...]

    turns = []
    for start_s, end_s, speaker_label in segments:
        a, b = int(start_s * sr), int(end_s * sr)
        chunk = wav[a:b]
        if len(chunk) < int(0.2 * sr):       # skip <0.2s blips
            continue
        out = asr.generate(input=chunk, fs=sr, language="en", use_itn=True)
        text = clean_text(out[0].get("text", "")) if out else ""
        if not text:
            continue
        turns.append({
            "speaker_id": speaker_label,       # real diarized cluster; match.py assigns names
            "start": round(start_s, 2),
            "end": round(end_s, 2),
            "text": text,
        })

    turns.sort(key=lambda t: t["start"])
    write_json(TURNS, turns)
    print(f"Got {len(turns)} turns (one per diarized segment).")


def clean_text(t: str) -> str:
    """SenseVoice emits rich tags like <|en|><|EMO_UNKNOWN|>. Strip them."""
    import re
    t = re.sub(r"<\|[^|]*\|>", "", t or "")
    return t.strip()


def _unused_flatten(res) -> list:
    """Kept for reference: paraformer sentence_info parsing."""
    turns = []
    idx = 0
    for item in res:
        sinfo = item.get("sentence_info")
        if sinfo:
            for s in sinfo:
                turns.append({
                    "speaker_id": f"Speaker_{s.get('spk', idx)}",
                    "start": round(s.get("start", 0) / 1000.0, 2),
                    "end": round(s.get("end", 0) / 1000.0, 2),
                    "text": clean_text(s.get("text")),
                })
                idx += 1
            continue

        # SenseVoice path: item has 'text' and optionally 'timestamp' (list of
        # [start_ms, end_ms] per token) — derive segment bounds from it.
        ts = item.get("timestamp")
        start = round(ts[0][0] / 1000.0, 2) if ts else 0.0
        end = round(ts[-1][1] / 1000.0, 2) if ts else 0.0
        turns.append({
            "speaker_id": f"Speaker_{idx}",
            "start": start,
            "end": end,
            "text": clean_text(item.get("text")),
        })
        idx += 1

    turns = [t for t in turns if t["text"]]
    turns.sort(key=lambda t: t["start"])
    return turns


if __name__ == "__main__":
    main()
