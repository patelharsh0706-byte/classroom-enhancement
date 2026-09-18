import { getScores } from "@/lib/data";
import { Card, SampleBanner, QualityPill } from "@/components/ui";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export default function StudentsView() {
  const { scores, isSample } = getScores();
  const maxTeam = Math.max(1, ...scores.teams.map((t) => t.total));

  return (
    <div>
      <h1 style={{ fontSize: 22, marginBottom: 6 }}>Students — Leaderboard</h1>
      <p style={{ color: "#8a90a2", marginTop: 0 }}>
        Quality beats volume. Group play, not a pure individual board — teams pull quiet voices in.
      </p>
      {isSample && <SampleBanner />}

      <Card title="🏆 Team standings">
        {scores.teams.length === 0 && (
          <p style={{ color: "#8a90a2", fontSize: 13, margin: 0, lineHeight: 1.6 }}>
            No teams configured. Add the group split to <code>teams</code> in{" "}
            <code>config.json</code> and re-run scoring — individual points below are already
            calculated and will roll up automatically.
          </p>
        )}
        {scores.teams.map((t) => (
          <div key={t.team} style={{ marginBottom: 14 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 14 }}>
              <span style={{ fontWeight: 600 }}>{t.team}</span>
              <span>{t.total} pts</span>
            </div>
            <div style={{ background: "#1d2230", borderRadius: 6, height: 10, marginTop: 4 }}>
              <div
                style={{
                  width: `${(100 * t.total) / maxTeam}%`,
                  background: "linear-gradient(90deg,#5ee0a0,#6fb1ff)",
                  height: 10,
                  borderRadius: 6,
                }}
              />
            </div>
            <div style={{ fontSize: 12, color: "#8a90a2", marginTop: 2 }}>{t.members.join(", ")}</div>
          </div>
        ))}
      </Card>

      <Card title="Individual breakdown">
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
          <thead>
            <tr style={{ textAlign: "left", color: "#8a90a2" }}>
              <th style={th}>#</th>
              <th style={th}>Student</th>
              <th style={th}>Points</th>
              <th style={th}>Turns</th>
              <th style={th}>Quality mix</th>
            </tr>
          </thead>
          <tbody>
            {scores.students.map((s, i) => (
              <tr key={s.name} style={{ borderTop: "1px solid #232838" }}>
                <td style={td}>{i + 1}</td>
                <td style={{ ...td, fontWeight: 600 }}>
                  {s.name}
                  {s.status === "unenrolled" && (
                    <span
                      title="Identified by human review only — this student has no voiceprint, so the system could not name them."
                      style={{ marginLeft: 8, fontSize: 11, color: "#d16a8a", fontWeight: 500 }}
                    >
                      ⚠ not enrolled
                    </span>
                  )}
                </td>
                <td style={td}>{s.total}</td>
                <td style={td}>{s.turns}</td>
                <td style={td}>
                  {Object.entries(s.by_quality).map(([q, n]) => (
                    <QualityPill key={q} label={q} count={n} />
                  ))}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
}

const th = { padding: "6px 8px", fontWeight: 500 };
const td = { padding: "8px" };
