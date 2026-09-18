import fs from "fs";
import path from "path";
import ExperimentWindow, { type ExpWindow } from "@/components/ExperimentWindow";

export const dynamic = "force-dynamic";
export const revalidate = 0;

function getWindows(): ExpWindow[] {
  try {
    const p = path.join(process.cwd(), "..", "data", "experiments.json");
    return (JSON.parse(fs.readFileSync(p, "utf-8")) as { windows: ExpWindow[] }).windows;
  } catch {
    return [];
  }
}

export default function ExperimentsView() {
  const windows = getWindows();
  const all = windows.flatMap((w) => w.turns);
  const confirmed = new Set(
    all.filter((t) => t.status === "confirmed" && !t.isProf).map((t) => t.name),
  );
  const unenrolled = new Set(all.filter((t) => t.status === "unenrolled").map((t) => t.name));

  return (
    <div>
      <h1 style={{ fontSize: 22, marginBottom: 6 }}>Verified speaker identification</h1>
      <p style={{ color: "#8a90a2", marginTop: 0, maxWidth: 780, lineHeight: 1.6 }}>
        Two windows from lecture 1, every speaker checked by a human. This is the only part of the
        system with ground truth behind it.
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, margin: "20px 0 24px" }}>
        <Stat value="8 / 8" label="Correct decisions" hint="7 named + 1 correct refusal" />
        <Stat value={String(confirmed.size)} label="Students identified" hint={[...confirmed].join(", ")} />
        <Stat value={String(unenrolled.size)} label="Unenrolled speaker" hint="no voiceprint on file" />
        <Stat value="0" label="False positives" hint="nobody wrongly named" />
      </div>

      <div
        style={{
          background: "#12211a", border: "1px solid #1f4535", color: "#9fe0bd",
          padding: "12px 16px", borderRadius: 8, fontSize: 13, marginBottom: 24, lineHeight: 1.6,
        }}
      >
        <b>Why window 2 matters.</b> Kaiting (0.34) and Rija (0.23) both scored <i>below</i> the
        original 0.536 threshold and were rejected as unreliable — then confirmed correct. That
        threshold was derived from phone-to-phone comparisons and applied to phone-to-room scores,
        which compresses everything downward. It was roughly 3x too high and was discarding real
        answers.
      </div>

      {windows.length === 0 ? (
        <p style={{ color: "#8a90a2" }}>
          No experiment data. Run <code>python3 scripts/build_experiments.py</code>.
        </p>
      ) : (
        windows.map((w) => <ExperimentWindow key={w.id} win={w} />)
      )}
    </div>
  );
}

function Stat({ value, label, hint }: { value: string; label: string; hint?: string }) {
  return (
    <div style={{ background: "#161a24", border: "1px solid #232838", borderRadius: 12, padding: 16 }}>
      <div style={{ fontSize: 26, fontWeight: 700 }}>{value}</div>
      <div style={{ fontSize: 13, color: "#8a90a2" }}>{label}</div>
      {hint && <div style={{ fontSize: 11, color: "#5a6072", marginTop: 3, lineHeight: 1.4 }}>{hint}</div>}
    </div>
  );
}
