"""Stage 5 — Scoring engine (plain Python).

Aggregates per-turn points into per-student totals, applies anti-gaming
discounts, and rolls up team scores. Anti-gaming is in from day one (per the
design): quality-weighting already happens in classify.py; here we add
volume/streak discounts so points can't be farmed.

Anti-gaming rules (from config.json):
  - Beyond `max_turns_per_window` turns inside `window_seconds`, extra turns by
    the same speaker are discounted by `overflow_multiplier`.
  - Consecutive same-quality turns by the same speaker are discounted by
    `consecutive_same_quality_discount` (repeating the same move = less reward).

Output: data/scores.json
    {
      "students": [ { "name", "total", "raw_total", "by_quality": {...}, "turns" } ],
      "teams":    [ { "team", "total", "members" } ]
    }

Run:  python3 pipeline/score.py
"""
from collections import defaultdict

from common import SCORED_TURNS, SCORES, load_config, read_json, write_json


def main() -> None:
    cfg = load_config()
    ag = cfg["anti_gaming"]
    teams_cfg = cfg.get("teams", {})

    turns = read_json(SCORED_TURNS)
    turns.sort(key=lambda t: (t["name"], t["start"]))

    # Track sliding-window counts + last quality, per student.
    window_starts = defaultdict(list)  # name -> list of recent turn start times
    last_quality = {}

    students = defaultdict(lambda: {
        "total": 0.0, "raw_total": 0, "by_quality": defaultdict(int), "turns": 0,
    })

    for t in turns:
        name = t["name"]
        pts = float(t["points"])
        raw = t["points"]

        # Volume discount: count turns within the trailing window.
        recent = [s for s in window_starts[name] if t["start"] - s <= ag["window_seconds"]]
        recent.append(t["start"])
        window_starts[name] = recent
        if len(recent) > ag["max_turns_per_window"]:
            pts *= ag["overflow_multiplier"]

        # Streak discount: same quality twice in a row by the same student.
        if last_quality.get(name) == t["quality"]:
            pts *= ag["consecutive_same_quality_discount"]
        last_quality[name] = t["quality"]

        s = students[name]
        s["total"] += pts
        s["raw_total"] += raw
        s["by_quality"][t["quality"]] += 1
        s["turns"] += 1

    student_list = []
    for name, s in students.items():
        student_list.append({
            "name": name,
            "total": round(s["total"], 2),
            "raw_total": s["raw_total"],
            "by_quality": dict(s["by_quality"]),
            "turns": s["turns"],
        })
    student_list.sort(key=lambda x: x["total"], reverse=True)

    # Team rollup (group play, not pure individual leaderboard).
    score_by_name = {s["name"]: s["total"] for s in student_list}
    team_list = []
    for team, members in teams_cfg.items():
        team_list.append({
            "team": team,
            "total": round(sum(score_by_name.get(m, 0) for m in members), 2),
            "members": members,
        })
    team_list.sort(key=lambda x: x["total"], reverse=True)

    write_json(SCORES, {"students": student_list, "teams": team_list})

    print("Final scores:")
    for s in student_list:
        print(f"  {s['name']}: {s['total']} ({s['turns']} turns, raw {s['raw_total']})")
    for t in team_list:
        print(f"  [{t['team']}] {t['total']}")


if __name__ == "__main__":
    main()
