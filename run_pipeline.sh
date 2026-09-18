#!/usr/bin/env bash
# Run the full pipeline on one lecture recording.
#
# Usage:
#   ./run_pipeline.sh                          # uses first .wav in audio/lectures/
#   ./run_pipeline.sh path/to/my_lecture.wav   # any wav file
#
# Prereqs: pip install -r requirements.txt, OPENAI_API_KEY in .env
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

LECTURE="${1:-}"

echo "== Stage 1/5: enroll =="
python3 pipeline/enroll.py

echo "== Stage 2/5: transcribe =="
python3 pipeline/transcribe.py $LECTURE

echo "== Stage 3/5: match =="
python3 pipeline/match.py $LECTURE

echo "== Stage 4/5: classify =="
python3 pipeline/classify.py

echo "== Stage 5/5: score =="
python3 pipeline/score.py

echo ""
echo "Done. Open http://localhost:3002 to see results."
