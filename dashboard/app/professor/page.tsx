import {
  getScores, getTimeline, getUnknownPct, getSourceAudio, getRoster, getReviewState,
} from "@/lib/data";
import { Card, SampleBanner, QualityPill, QUALITY_COLORS } from "@/components/ui";
import SyncedTimeline from "@/components/SyncedTimeline";
import LearnButton from "@/components/LearnButton";

// The pipeline rewrites ../data/*.json between runs, so this page must re-read
// them per request instead of serving a build-time snapshot.
export const dynamic = "force-dynamic";
export const revalidate = 0;

export default function ProfessorView() {
  const { scores, isSample } = getScores();
  const { turns: timeline } = getTimeline();
  const unknownPct = getUnknownPct();
  const audioFile = getSourceAudio();

  // Class-wide quality distribution.
  const dist: Record<string, number> = {};
  for (const s of scores.students) {
    for (const [q, n] of Object.entries(s.by_quality)) dist[q] = (dist[q] ?? 0) + n;
  }
  const totalTurns = Object.values(dist).reduce((a, b) => a + b, 0);

  // Talk share is computed from the pre-filter layer, so it counts the
  // professor — unlike the scoring path, which drops him by design.
  const secs = (only: boolean) =>
    timeline.filter((t) => t.isProf === only).reduce((a, t) => a + (t.end - t.start), 0);
  const profSecs = secs(true);
  const studentSecs = secs(false);
  const talkShare =
    profSecs + studentSecs > 0
      ? Math.round((100 * studentSecs) / (profSecs + studentSecs))
      : null;

  return (
    <div>
      <h1 style={{ fontSize: 22, marginBottom: 6 }}>Professor — Engagement Health</h1>
      <p style={{ color: "#8a90a2", marginTop: 0 }}>
        Real-time read on who contributed and how — the feedback loop you don&apos;t get until exam time.
      </p>
      {isSample && <SampleBanner />}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 16, marginBottom: 20 }}>
        <Stat label="Participating students" value={String(scores.students.length)} />
        <Stat label="Scored turns" value={String(totalTurns)} />
        <Stat
          label="Unknown (matching gap)"
          value={unknownPct === null ? "—" : `${unknownPct}%`}
          hint="honesty metric"
        />
        <Stat
          label="Student talk share"
          value={talkShare === null ? "—" : `${talkShare}%`}
          hint="of spoken time, vs professor"
        />
      </div>

      <Card title="Quality distribution (whole class)">
        {totalTurns === 0 ? (
          <p style={{ color: "#8a90a2" }}>No turns yet.</p>
        ) : (
          <div>
            <div style={{ display: "flex", height: 22, borderRadius: 6, overflow: "hidden" }}>
              {Object.entries(dist).map(([q, n]) => (
                <div
                  key={q}
                  title={`${q}: ${n}`}
                  style={{ width: `${(100 * n) / totalTurns}%`, background: QUALITY_COLORS[q] ?? "#888" }}
                />
              ))}
            </div>
            <div style={{ marginTop: 12 }}>
              {Object.entries(dist).map(([q, n]) => (
                <QualityPill key={q} label={q} count={n} />
              ))}
            </div>
          </div>
        )}
      </Card>

      <LearnButton pendingCount={getReviewState().size} />

      <Card title="Turn timeline — follows the recording">
        <SyncedTimeline
          initialTurns={timeline}
          initialFilename={audioFile}
          roster={getRoster()}
        />
      </Card>
    </div>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div style={{ background: "#161a24", border: "1px solid #232838", borderRadius: 12, padding: 16 }}>
      <div style={{ fontSize: 28, fontWeight: 700 }}>{value}</div>
      <div style={{ fontSize: 13, color: "#8a90a2" }}>{label}</div>
      {hint && <div style={{ fontSize: 11, color: "#5a6072", marginTop: 2 }}>{hint}</div>}
    </div>
  );
}

