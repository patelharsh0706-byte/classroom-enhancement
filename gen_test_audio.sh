#!/usr/bin/env bash
# Generate synthetic test audio using macOS `say` (different voice per speaker)
# so the pipeline can be smoke-tested without recording real audio.
#
#   Priya → Rishi (en_IN)   Arjun → Daniel (en_GB)
#   Rajan → Karen (en_AU)   Prof  → Fred (en_US)
#
# Usage:  ./gen_test_audio.sh
set -euo pipefail
cd "$(dirname "$0")"

mk() {  # mk <voice> <outfile.wav> <text>
  local voice="$1" out="$2" text="$3" tmp
  tmp="$(mktemp -t saytmp).aiff"
  say -v "$voice" -o "$tmp" "$text"
  afconvert "$tmp" "$out" -d LEI16@16000 -c 1 -f WAVE
  rm -f "$tmp"
  echo "  wrote $out"
}

mkdir -p audio/enrollment audio/lectures

echo "== enrollment clips =="
mk Rishi  audio/enrollment/priya_1.wav "Hi, I'm Priya. I'm a computer science student at NUS."
mk Rishi  audio/enrollment/priya_2.wav "Hello, this is Priya speaking, testing the microphone."
mk Daniel audio/enrollment/arjun_1.wav "Hi, I'm Arjun. I study machine learning at NUS."
mk Daniel audio/enrollment/arjun_2.wav "Hello, this is Arjun speaking, just a quick test."
mk Karen  audio/enrollment/rajan_1.wav "Hi, I'm Rajan. I'm in the data science program."
mk Karen  audio/enrollment/rajan_2.wav "Hello, this is Rajan speaking into the microphone."

echo "== mock lecture (single track, speakers in sequence) =="
# One file, speakers concatenated, mimicking a discussion. Diarization should
# segment it back into distinct speakers.
LECT_TXT="audio/lectures/_lecture_script.txt"
: > "$LECT_TXT"
build_lecture() {
  local tmpdir; tmpdir="$(mktemp -d)"
  local i=0
  add() {  # add <voice> <text>
    local f="$tmpdir/$(printf '%03d' $i).aiff"
    say -v "$1" -o "$f" "$2"
    echo "$2" >> "$LECT_TXT"
    i=$((i+1))
  }
  add Fred   "Today we will discuss the attention mechanism in transformers."
  add Rishi  "This feels similar to convolutional networks, where locality is a built-in bias. But here we are removing that bias entirely and letting the model attend to everything."
  add Fred   "Exactly. So how is the attention distribution actually computed?"
  add Arjun  "Are the query and key matrices learned weights, or are they fixed?"
  add Fred   "They are learned. Priya, can you build on that?"
  add Rishi  "So the softmax over query times key transpose gives the attention weights, and those weight the values."
  add Rajan  "Okay, got it."
  add Arjun  "Does that mean self attention has quadratic cost in the sequence length?"
  add Rajan  "I agree, that seems expensive."
  # concatenate all clips into one wav
  local list="$tmpdir/list.txt"
  for f in "$tmpdir"/*.aiff; do echo "file '$f'"; done > "$list"
  # afconvert each to wav then cat via sox-free approach: use ffmpeg if present, else afconvert+python
  if command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg -y -f concat -safe 0 -i "$list" -ar 16000 -ac 1 audio/lectures/lecture_01.wav >/dev/null 2>&1
  else
    # afconvert can't concat; stitch with python + soundfile
    python3 - "$tmpdir" <<'PY'
import sys, glob, os, subprocess, numpy as np, soundfile as sf
d = sys.argv[1]
chunks = []
for aiff in sorted(glob.glob(os.path.join(d, "*.aiff"))):
    wav = aiff.replace(".aiff", ".wav")
    subprocess.run(["afconvert", aiff, wav, "-d", "LEI16@16000", "-c", "1", "-f", "WAVE"], check=True)
    data, sr = sf.read(wav)
    chunks.append(data.astype(np.float32))
    chunks.append(np.zeros(int(0.7 * sr), dtype=np.float32))  # 0.7s gap between turns
out = np.concatenate(chunks)
sf.write("audio/lectures/lecture_01.wav", out, 16000, subtype="PCM_16")
print("  stitched lecture_01.wav via soundfile")
PY
  fi
  rm -rf "$tmpdir"
}
build_lecture
echo "  wrote audio/lectures/lecture_01.wav"
echo "Done."
