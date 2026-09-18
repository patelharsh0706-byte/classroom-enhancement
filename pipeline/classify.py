"""Stage 4 — Quality classification (gpt-4o-mini via OpenAI).

Classifies each student turn into a participation-quality label. The
professor's own turns and Unknown turns are skipped.

Output: data/scored_turns.json
    [ { "name": "Priya", "text": "...", "quality": "synthesizing", "points": 3 }, ... ]

Run:  python3 pipeline/classify.py  (OPENAI_API_KEY must be in .env or environment)
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from common import NAMED_TURNS, SCORED_TURNS, load_config, read_json, write_json

LABELS = ["synthesizing", "answering", "clarifying", "off-topic", "acknowledgement"]
MODEL = "gpt-4o-mini"

SYSTEM = (
    "You classify a single student turn from a university lecture discussion by its "
    "participation quality. Definitions:\n"
    "- synthesizing: connects ideas, draws an analogy, builds a new insight\n"
    "- answering: gives a substantive answer to a posed question\n"
    "- clarifying: asks a genuine question to understand the material\n"
    "- off-topic: unrelated to the lecture\n"
    "- acknowledgement: low-content filler ('okay', 'got it', 'I agree')\n\n"
    'Return ONLY a JSON object: {"label": "<label>"}'
)


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not found. Add it to .env or export it.")

    from openai import OpenAI
    client = OpenAI()

    cfg = load_config()
    weights = cfg["quality_weights"]
    prof = cfg["professor_label"]
    named = read_json(NAMED_TURNS)

    scored = []
    for t in named:
        if t["name"] == prof:
            continue  # professor turns are never student participation
        # Unknown turns: we have the text, so classify quality anyway.
        # They'll show in the dashboard as "Unidentified" — honest but still useful.
        label = classify_turn(client, t["text"])
        scored.append({
            "name": t["name"],
            "start": t["start"],
            "end": t["end"],
            "text": t["text"],
            "quality": label,
            "points": weights.get(label, 0),
        })
        print(f"  {t['name']}: {label} (+{weights.get(label, 0)}) — {t['text'][:55]}")

    write_json(SCORED_TURNS, scored)
    print(f"Classified {len(scored)} student turns.")


def classify_turn(client, text: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Turn: {text}"},
        ],
    )
    try:
        label = json.loads(resp.choices[0].message.content).get("label", "off-topic")
        return label if label in LABELS else "off-topic"
    except Exception:
        return "off-topic"


if __name__ == "__main__":
    main()
