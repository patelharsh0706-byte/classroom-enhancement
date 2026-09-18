"""Speaker-change-point detection via pyannote.audio 3.1.

Returns real diarized speaker clusters (reused across all of that speaker's
turns in the file) — NOT silence-bounded VAD chunks. This is what lets two
people who talk with <1s gaps between them still get split into separate
speakers, which a silence-threshold VAD fundamentally cannot do.

Setup (one-time, manual, on huggingface.co):
  1. Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1
  2. Accept terms: https://huggingface.co/pyannote/segmentation-3.0
  3. Create a read token: https://huggingface.co/settings/tokens
  4. Add HF_TOKEN=hf_xxx to .env
"""
import math
import os
import sys
from pathlib import Path

from common import pick_device_pyannote


def diarize(audio_path: str, chunk_seconds: float = 0, batch_size: int = 8) -> list:
    """Returns [(start_s, end_s, "SPEAKER_00"), ...] sorted by start.

    chunk_seconds > 0 diarizes the file in fixed-length windows instead of one
    pass. Single-pass holds a segmentation score, a speaker embedding, and an
    all-pairs similarity matrix for every window in the file alive at once, so
    memory scales with total duration: on an 8 GB Mac, 89 minutes of lecture ran
    1h43m while burning only 12 min of CPU — the rest was swap page-ins.
    Chunking caps peak memory at one chunk's worth.

    Speaker labels are chunk-local and prefixed ("c00_SPEAKER_01"): pyannote
    clusters each chunk independently, so SPEAKER_00 in one chunk is not the
    same person as SPEAKER_00 in the next. That is safe here only because
    match.py maps every cluster to a real name via the enrolled voiceprints —
    identity is recovered per cluster, and score.py groups by name, so a
    student's turns from different chunks reconcile automatically. The cost is
    that brief participation leaves a small cluster with little audio to embed,
    which lowers match confidence (pushing turns to Unknown rather than to a
    wrong name).
    """
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit(
            "HF_TOKEN not found in .env or environment.\n"
            "pyannote/speaker-diarization-3.1 is gated — you must:\n"
            "  1. Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  2. Accept terms: https://huggingface.co/pyannote/segmentation-3.0\n"
            "  3. Create a read token: https://huggingface.co/settings/tokens\n"
            "  4. Add HF_TOKEN=hf_xxx to .env"
        )

    import huggingface_hub
    _orig_hf_hub_download = huggingface_hub.hf_hub_download
    def _hf_hub_download_compat(*args, **kwargs):
        # pyannote.audio 3.1.1 calls hf_hub_download(..., use_auth_token=...),
        # a kwarg newer huggingface_hub renamed to `token`.
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return _orig_hf_hub_download(*args, **kwargs)
    huggingface_hub.hf_hub_download = _hf_hub_download_compat

    import numpy as np
    if not hasattr(np, "NaN"):
        # pyannote.audio 3.1.1 references the legacy np.NaN/np.NAN aliases,
        # removed in NumPy 2.0 in favor of np.nan. Same value, restoring names.
        np.NaN = np.nan
    if not hasattr(np, "NAN"):
        np.NAN = np.nan

    import torchaudio
    if not hasattr(torchaudio, "set_audio_backend"):
        # pyannote.audio 3.1.1 calls these obsolete torchaudio APIs; newer
        # torchaudio (>=2.1) selects backends automatically and removed them.
        # No-op shims — safe since we never rely on explicit backend choice.
        torchaudio.set_audio_backend = lambda backend: None
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"

    import sys
    import types
    if "torchaudio.backend" not in sys.modules:
        # torchaudio>=2.9 deleted the torchaudio.backend.* submodule tree.
        # pyannote.audio 3.1.1's segmentation task code does
        # `from torchaudio.backend.common import AudioMetaData` purely to
        # type-annotate a return value — it never constructs or reads fields
        # off the object — so a bare placeholder class is sufficient.
        backend_mod = types.ModuleType("torchaudio.backend")
        common_mod = types.ModuleType("torchaudio.backend.common")

        class AudioMetaData:
            def __init__(self, sample_rate=16000, num_frames=0, num_channels=1,
                         bits_per_sample=16, encoding="PCM_S"):
                self.sample_rate = sample_rate
                self.num_frames = num_frames
                self.num_channels = num_channels
                self.bits_per_sample = bits_per_sample
                self.encoding = encoding

        common_mod.AudioMetaData = AudioMetaData
        backend_mod.common = common_mod
        sys.modules["torchaudio.backend"] = backend_mod
        sys.modules["torchaudio.backend.common"] = common_mod
        torchaudio.backend = backend_mod

    from pyannote.audio import Pipeline
    import torch

    _orig_torch_load = torch.load
    def _torch_load_compat(*args, **kwargs):
        # PyTorch >=2.6 flipped torch.load's default to weights_only=True;
        # pyannote.audio 3.1.1's checkpoints pickle a torch.torch_version
        # .TorchVersion object, which that safe loader rejects. These are
        # official pyannote/HF checkpoints, so weights_only=False is safe here.
        # lightning_fabric explicitly passes weights_only=None (not omitted),
        # so setdefault wouldn't catch it — force the override unconditionally.
        if kwargs.get("weights_only") is not False:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _torch_load_compat

    torch.set_num_threads(os.cpu_count() or 4)

    device = pick_device_pyannote()
    print(f"Diarize — device={device}  file={audio_path}")

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token,
        )
    except Exception as e:
        sys.exit(
            f"Failed to load pyannote/speaker-diarization-3.1: {e}\n"
            "Likely cause: HF_TOKEN invalid, or gated terms not accepted for "
            "BOTH speaker-diarization-3.1 and segmentation-3.0."
        )
    finally:
        torch.load = _orig_torch_load

    pipeline.to(torch.device(device))

    # pyannote.audio defaults both batch sizes to 1 — i.e. it runs the
    # sliding-window segmentation and embedding models one window at a time.
    # Batching lets PyTorch vectorize across windows instead of looping in
    # Python, which is a large wall-clock win even on CPU. Each in-flight window
    # costs memory though, so on a small-RAM machine this trades against the
    # very thing chunking is fixing — keep it modest.
    pipeline.segmentation_batch_size = batch_size
    pipeline.embedding_batch_size = batch_size

    # Don't hand pipeline() a bare path — internally it calls torchaudio.load(),
    # which on this torchaudio version requires the separate torchcodec package
    # (the exact codec dependency load_wav_mono16k in common.py already avoids
    # via soundfile). Passing a pre-loaded waveform dict skips that code path.
    from common import load_wav_mono16k
    waveform, sr = load_wav_mono16k(audio_path)
    total_s = waveform.shape[1] / sr

    if not chunk_seconds or chunk_seconds <= 0 or total_s <= chunk_seconds:
        return _run(pipeline, waveform, sr, offset=0.0, prefix="")

    import gc

    segments = []
    n_chunks = math.ceil(total_s / chunk_seconds)
    for i in range(n_chunks):
        start_s = i * chunk_seconds
        end_s = min((i + 1) * chunk_seconds, total_s)
        a, b = int(start_s * sr), int(end_s * sr)
        print(f"  chunk {i + 1}/{n_chunks}  [{start_s / 60:.1f}–{end_s / 60:.1f} min]")
        # .clone() so the slice doesn't keep the full-file tensor alive via a view.
        chunk = waveform[:, a:b].clone()
        segments.extend(_run(pipeline, chunk, sr, offset=start_s, prefix=f"c{i:02d}_"))
        del chunk
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    segments.sort(key=lambda s: s[0])
    print(f"Pyannote found {len(segments)} segments across "
          f"{len({s[2] for s in segments})} chunk-local speaker clusters.")
    return segments


def _run(pipeline, waveform, sr, offset: float, prefix: str) -> list:
    """Diarize one waveform, shifting times by `offset` back onto the full-file
    timeline and namespacing labels with `prefix` so chunk-local cluster ids
    from different chunks can never collide."""
    diarization = pipeline({"waveform": waveform, "sample_rate": sr})
    segments = [
        (round(turn.start + offset, 3), round(turn.end + offset, 3), f"{prefix}{label}")
        for turn, _, label in diarization.itertracks(yield_label=True)
    ]
    segments.sort(key=lambda s: s[0])
    if not prefix:
        print(f"Pyannote found {len(segments)} segments across "
              f"{len({s[2] for s in segments})} speakers.")
    return segments
