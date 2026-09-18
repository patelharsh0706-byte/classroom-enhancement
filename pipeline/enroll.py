"""Stage 1 — Enrollment.

Build a voiceprint database from short enrollment clips. For each student we
average the ECAPA-TDNN embeddings of all their clips into one reference vector.

Expected input layout:
    audio/enrollment/<name>_1.wav
    audio/enrollment/<name>_2.wav   (the prefix before the first "_" is the name)

Output: data/voiceprints.json  ->  { "Priya": [192 floats], "Arjun": [...] }

Run:  python3 pipeline/enroll.py
"""
import json
import sys
from collections import defaultdict

import numpy as np

from common import (
    AUDIO_ENROLL, VOICEPRINTS, MODELS, NAME_MAP, write_json, pick_device_speechbrain,
)


def name_from_filename(path) -> str:
    # "priya_1.wav" -> "priya" ; capitalize for display
    stem = path.stem
    base = stem.split("_")[0]
    # Real names aren't all single tokens (some are three words), and a
    # filename can't hold spaces or underscores here. apply_learning.py writes a
    # slug -> display-name map so those round-trip. Consulting it HERE rather
    # than post-processing voiceprints.json means a plain `python3
    # pipeline/enroll.py` always produces the same keys as the learning path.
    if NAME_MAP.exists():
        try:
            mapped = json.loads(NAME_MAP.read_text()).get(base)
            if mapped:
                return mapped
        except Exception:
            pass  # a broken map must not block enrollment
    return base.capitalize()


def main() -> None:
    device = pick_device_speechbrain()
    print(f"Enrollment — device={device}")

    clips = sorted(AUDIO_ENROLL.glob("*.wav"))
    if not clips:
        sys.exit(
            f"No enrollment clips found in {AUDIO_ENROLL}.\n"
            "Add files named <name>_1.wav, <name>_2.wav, ... first."
        )

    # Lazy import so the script can at least print the help above without torch.
    from speechbrain.inference.speaker import EncoderClassifier

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(MODELS / "ecapa"),
        run_opts={"device": device},
    )

    by_name = defaultdict(list)
    skipped = []
    for clip in clips:
        name = name_from_filename(clip)
        try:
            emb = embed_file(encoder, clip)
        except Exception as e:
            # A clip can be unusable even when the container looks valid — e.g. a
            # phone upload whose duration header is intact but which holds zero
            # audio packets. Skip it rather than losing the whole enrollment run.
            skipped.append((clip.name, f"{type(e).__name__}: {e}"))
            print(f"  SKIPPED {clip.name} -> {name}  ({type(e).__name__})")
            continue
        by_name[name].append(emb)
        print(f"  embedded {clip.name} -> {name}")

    voiceprints = {}
    for name, embs in by_name.items():
        mean = np.mean(np.stack(embs), axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-9)  # L2-normalize reference
        voiceprints[name] = mean.tolist()

    write_json(VOICEPRINTS, voiceprints)
    print(f"Enrolled {len(voiceprints)} speakers: {', '.join(voiceprints)}")
    if skipped:
        print(f"\n{len(skipped)} clip(s) skipped — these speakers have NO voiceprint "
              f"and will fall to Unknown:")
        for fname, why in skipped:
            print(f"  - {fname}  ({why})")


# ECAPA's conv stack pads by 2 frames, so anything under ~5 frames of mel input
# (~0.1s) crashes rather than returning a weak embedding.
MIN_SECONDS = 0.5


def embed_file(encoder, path) -> np.ndarray:
    from common import load_wav_mono16k

    signal, sr = load_wav_mono16k(path)  # [1, N] mono 16k
    if signal.shape[1] < int(MIN_SECONDS * sr):
        raise ValueError(
            f"only {signal.shape[1] / sr:.3f}s of decodable audio "
            f"(need >= {MIN_SECONDS}s) — file is empty or corrupt"
        )
    emb = encoder.encode_batch(signal).squeeze().detach().cpu().numpy()
    return emb


if __name__ == "__main__":
    main()
