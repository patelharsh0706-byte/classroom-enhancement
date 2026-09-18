// Reads the pipeline's JSON outputs (../data/) on the server.
// Falls back to a baked-in sample so the dashboard renders before the
// pipeline has ever run (useful for the demo / first look).
import fs from "fs";
import path from "path";

export type ScoredTurn = {
  name: string;
  start: number;
  end: number;
  text: string;
  quality: string;
  points: number;
};

export type Student = {
  name: string;
  total: number;
  raw_total: number;
  by_quality: Record<string, number>;
  turns: number;
  /** "unenrolled" = identified by human review only; no voiceprint on file. */
  status?: "confirmed" | "unenrolled" | "unverified";
};

export type Team = { team: string; total: number; members: string[] };

export type Scores = { students: Student[]; teams: Team[] };

const DATA_DIR = path.join(process.cwd(), "..", "data");

function readOr<T>(file: string, fallback: T): T {
  try {
    return JSON.parse(fs.readFileSync(path.join(DATA_DIR, file), "utf-8")) as T;
  } catch {
    return fallback;
  }
}

const SAMPLE_SCORES: Scores = {
  students: [
    { name: "Priya", total: 8, raw_total: 8, by_quality: { synthesizing: 2, clarifying: 1 }, turns: 3 },
    { name: "Arjun", total: 4, raw_total: 5, by_quality: { answering: 1, clarifying: 1, acknowledgement: 1 }, turns: 3 },
    { name: "Rajan", total: 2, raw_total: 2, by_quality: { clarifying: 1, "off-topic": 1 }, turns: 2 },
  ],
  teams: [
    { team: "Team A", total: 12, members: ["Priya", "Arjun"] },
    { team: "Team B", total: 2, members: ["Rajan"] },
  ],
};

const SAMPLE_TURNS: ScoredTurn[] = [
  { name: "Priya", start: 12.4, end: 19.2, text: "This feels like the CNN locality bias, but here we drop it entirely.", quality: "synthesizing", points: 3 },
  { name: "Arjun", start: 25.1, end: 28.0, text: "Are Q and K learned weight matrices?", quality: "clarifying", points: 2 },
  { name: "Rajan", start: 40.0, end: 41.5, text: "Okay, got it.", quality: "acknowledgement", points: 0 },
  { name: "Priya", start: 55.0, end: 61.0, text: "So softmax over QK^T gives the attention distribution.", quality: "answering", points: 2 },
];

export function getScores(): { scores: Scores; isSample: boolean } {
  const scores = readOr<Scores | null>("scores.json", null);
  if (scores) return { scores, isSample: false };
  return { scores: SAMPLE_SCORES, isSample: true };
}

export function getTurns(): { turns: ScoredTurn[]; isSample: boolean } {
  const turns = readOr<ScoredTurn[] | null>("scored_turns.json", null);
  if (turns) return { turns, isSample: false };
  return { turns: SAMPLE_TURNS, isSample: true };
}

export function getSourceAudio(): string | null {
  const record = readOr<{ filename: string } | null>("source_audio.json", null);
  return record?.filename ?? null;
}

// Honesty metric: % of turns left Unknown by speaker matching.
// Read from named_turns.json if present (the un-scored, pre-filter layer).
export function getUnknownPct(): number | null {
  const named = readOr<{ name: string }[] | null>("named_turns.json", null);
  if (!named || named.length === 0) return null;
  const unknown = named.filter((t) => t.name === "Unknown").length;
  return Math.round((100 * unknown) / named.length);
}

export type Candidate = { name: string; sim: number; z: number | null };

export type ReviewFlag = {
  needed: boolean;
  reason: "low-sim" | "low-z" | "confusable" | "unknown" | "audit";
  confusableWith?: string;
};

export type HumanVerdict = {
  name: string;
  verdict: "confirm" | "correct" | "unenrolled" | "unknown";
  ts: string;
};

export type TimelineTurn = {
  name: string;
  start: number;
  end: number;
  text: string;
  isProf: boolean;
  quality?: string;
  points?: number;
  clusterId?: string;
  candidates?: Candidate[];
  sim?: number;
  review?: ReviewFlag;
  human?: HumanVerdict;
  /**
   * `start`/`end` are positions in whatever audio file the dashboard is playing —
   * which is a stitched clip of just the analysed windows, not the 90-minute
   * lecture. absLabel preserves the real lecture time for display.
   */
  absLabel?: string;
};

type NamedTurn = {
  name: string;
  start: number;
  end: number;
  text: string;
  cluster_id?: string;
  candidates?: Candidate[];
  confidence?: number;
  abs_label?: string;
};

export type GroundTruthRecord = {
  id: string;
  ts: string;
  lecture: string;
  cluster_id: string;
  verdict: HumanVerdict["verdict"];
  corrected_name: string;
  system_guess?: string;
  system_sim?: number;
};

type ClusterFile = {
  clusters: Record<string, {
    cluster_id: string;
    match_status: "accepted" | "review" | "unknown";
    audit_sampled?: boolean;
    candidates: Candidate[];
    spans: [number, number][];
    n_turns: number;
    speech_seconds: number;
  }>;
};

type Confusables = { by_name: Record<string, { name: string; sim: number }[]> };

export function getClusters(): ClusterFile | null {
  return readOr<ClusterFile | null>("clusters.json", null);
}

export function getConfusables(): Confusables | null {
  return readOr<Confusables | null>("confusables.json", null);
}

export function getRoster(): string[] {
  try {
    const cfg = JSON.parse(
      fs.readFileSync(path.join(process.cwd(), "..", "config.json"), "utf-8"),
    ) as { roster?: string[] };
    return cfg.roster ?? [];
  } catch {
    return [];
  }
}

/**
 * Replay the append-only verdict log. Last write wins per cluster, so a
 * professor changing their mind is just a newer line — nothing is ever
 * rewritten or compacted, and the full decision history stays auditable.
 */
export function getReviewState(): Map<string, GroundTruthRecord> {
  const out = new Map<string, GroundTruthRecord>();
  try {
    const raw = fs.readFileSync(path.join(DATA_DIR, "ground_truth.jsonl"), "utf-8");
    for (const line of raw.split("\n")) {
      if (!line.trim()) continue;
      try {
        const rec = JSON.parse(line) as GroundTruthRecord;
        if (rec.cluster_id) out.set(rec.cluster_id, rec);
      } catch {
        /* skip a malformed line rather than losing the whole log */
      }
    }
  } catch {
    /* no corrections yet */
  }
  return out;
}

/**
 * Merged two-lane timeline, read fresh from disk on every call.
 *
 * The professor's turns exist in named_turns.json but are deliberately dropped
 * from scored_turns.json (classify.py skips them — teaching is not student
 * participation). The timeline needs both lanes, so read the pre-filter layer
 * and join the quality labels back on by (name, start).
 *
 * No sample fallback here on purpose: this view is the transcript, and an
 * invented transcript shown next to a real recording would be actively
 * misleading. Empty means the pipeline has not produced turns yet.
 */
export function getTimeline(): { turns: TimelineTurn[]; profLabel: string } {
  const profLabel = getProfessorLabel();
  const named = readOr<NamedTurn[] | null>("named_turns.json", null);
  if (!named || named.length === 0) return { turns: [], profLabel };

  const scored = readOr<ScoredTurn[]>("scored_turns.json", []);
  const byKey = new Map(scored.map((s) => [`${s.name}@${s.start}`, s]));
  const byStart = new Map(scored.map((s) => [s.start, s]));
  const verdicts = getReviewState();
  const clusters = getClusters()?.clusters ?? {};
  const confusable = getConfusables()?.by_name ?? {};

  const turns = named.map((t) => {
    const cid = t.cluster_id;
    const verdict = cid ? verdicts.get(cid) : undefined;

    // 1. A human verdict overrides the model immediately — no pipeline re-run.
    //    Below the accept threshold the model's guess is NOT shown as a name:
    //    it displays as "Unknown" with the candidates offered as choices. A
    //    sub-threshold guess is not an identification — confirmed 2026-08-31,
    //    when a 0.143 "Richard" was actually Student-Z, who has no voiceprint at
    //    all. Showing the guess as a name invites it to be believed.
    const cluster0 = cid ? clusters[cid] : undefined;
    const unresolved = !verdict && cluster0 !== undefined && cluster0.match_status !== "accepted";
    const name = verdict ? verdict.corrected_name : unresolved ? "Unknown" : t.name;

    // 2. Join quality AFTER the override. The join key contains the name, so
    //    joining first would silently drop the label off every corrected turn.
    //    Fall back to start-only, which is unique within a lecture anyway.
    const hit = byKey.get(`${name}@${t.start}`) ?? byStart.get(t.start);

    // 3. Does this need a human? Suppressed once a human has ruled on it.
    const cluster = cluster0;
    const cands = t.candidates ?? cluster?.candidates ?? [];
    const review = verdict ? { needed: false, reason: "audit" as const } : reviewFlag(cluster, cands, confusable);

    return {
      name,
      start: t.start,
      end: t.end,
      text: t.text,
      isProf: name === profLabel, // effective name, so a relabel moves lanes
      quality: hit?.quality,
      points: hit?.points,
      clusterId: cid,
      candidates: cands.slice(0, 3),
      sim: t.confidence,
      absLabel: t.abs_label,
      review,
      human: verdict
        ? { name: verdict.corrected_name, verdict: verdict.verdict, ts: verdict.ts }
        : undefined,
    };
  });
  turns.sort((a, b) => a.start - b.start);
  return { turns, profLabel };
}

function reviewFlag(
  cluster: ClusterFile["clusters"][string] | undefined,
  cands: Candidate[],
  confusable: Record<string, { name: string; sim: number }[]>,
): ReviewFlag {
  const status = cluster?.match_status;
  if (status === "unknown") return { needed: true, reason: "unknown" };
  if (status === "review") return { needed: true, reason: "low-sim" };

  // Confident but ambiguous: the top two are a known-confusable pair, so the
  // runner-up's score is leakage rather than an independent hypothesis. A
  // similarity threshold cannot catch this case.
  if (cands.length >= 2) {
    const pair = confusable[cands[0].name]?.find((c) => c.name === cands[1].name);
    if (pair) return { needed: true, reason: "confusable", confusableWith: cands[1].name };
  }

  // A sample of confident clusters is reviewed too, so the ground-truth store
  // contains high-similarity examples. Without these, any threshold later
  // fitted to the store sees only hard cases and is unrepairably optimistic.
  if (cluster?.audit_sampled) return { needed: true, reason: "audit" };

  return { needed: false, reason: "audit" };
}

function getProfessorLabel(): string {
  try {
    const cfg = JSON.parse(
      fs.readFileSync(path.join(process.cwd(), "..", "config.json"), "utf-8"),
    ) as { professor_label?: string };
    return cfg.professor_label ?? "Prof";
  } catch {
    return "Prof";
  }
}
