"""Shared paths + config + JSON helpers for the pipeline.

Every stage reads/writes a well-defined JSON file in data/. Keeping these
interfaces clean is what lets the plumbing (flat JSON now) be swapped for a
queue/DB later without rewriting any stage logic.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO_ENROLL = ROOT / "audio" / "enrollment"
AUDIO_LECTURES = ROOT / "audio" / "lectures"
DATA = ROOT / "data"
MODELS = ROOT / "models"

# Pipeline artifacts (one per stage)
VOICEPRINTS = DATA / "voiceprints.json"
TURNS = DATA / "turns.json"
NAMED_TURNS = DATA / "named_turns.json"
SCORED_TURNS = DATA / "scored_turns.json"
SCORES = DATA / "scores.json"
SOURCE_AUDIO = DATA / "source_audio.json"

# Per-cluster identity decisions + ranked candidates — what the dashboard's
# correction picker reads. Separate from named_turns.json because identity is a
# property of a speaker cluster, not of an individual turn.
CLUSTERS = DATA / "clusters.json"
CONFUSABLES = DATA / "confusables.json"
# Append-only human verdicts. The only thing allowed to enrich a voiceprint.
GROUND_TRUTH = DATA / "ground_truth.jsonl"
NAME_MAP = DATA / "name_map.json"


def load_config() -> dict:
    with open(ROOT / "config.json") as f:
        return json.load(f)


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"  → wrote {path.relative_to(ROOT)}")


def pick_device() -> str:
    """Apple Silicon -> mps, NVIDIA -> cuda, else cpu."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_wav_mono16k(path):
    """Load a WAV as a mono 16 kHz torch tensor shaped [1, samples].

    Uses soundfile (not torchaudio.load) — torchaudio 2.11 requires a separate
    torchcodec package to decode, which we avoid. Resampling, if needed, uses
    torchaudio.functional.resample (a pure op, no codec dependency).
    Returns (tensor[1, N] float32, sr=16000).
    """
    import numpy as np
    import soundfile as sf
    import torch

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # [N, channels]
    mono = data.mean(axis=1)  # downmix to mono -> [N]
    sig = torch.from_numpy(np.ascontiguousarray(mono)).unsqueeze(0)  # [1, N]
    if sr != 16000:
        import torchaudio
        sig = torchaudio.functional.resample(sig, sr, 16000)
        sr = 16000
    return sig, sr


def pick_device_speechbrain() -> str:
    """SpeechBrain 1.1.0 doesn't handle MPS (it only sets device_type for
    cpu/cuda, then crashes), so never hand it 'mps'. The ECAPA embedding is
    tiny — CPU is plenty fast for short clips."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def pick_device_pyannote() -> str:
    """Verified 2026-06-16: pyannote 3.1 on this Mac's MPS produces identical
    segment/speaker output to CPU (22 segments, 2 speakers on podcast_01.wav)
    in ~1/5th the wall time (83s vs 382s) — unlike SpeechBrain's ECAPA, which
    actually does crash on MPS (see pick_device_speechbrain()), so the two
    don't share a rationale despite the similar-looking fallback chain."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
