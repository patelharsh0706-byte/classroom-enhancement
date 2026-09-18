import type { ReactNode } from "react";

export const QUALITY_COLORS: Record<string, string> = {
  synthesizing: "#5ee0a0",
  answering: "#6fb1ff",
  clarifying: "#c79bff",
  "off-topic": "#5a6072",
  acknowledgement: "#8a90a2",
};

export function Card({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section
      style={{
        background: "#161a24",
        border: "1px solid #232838",
        borderRadius: 12,
        padding: 20,
        marginBottom: 20,
      }}
    >
      <h2 style={{ margin: "0 0 14px", fontSize: 16, color: "#c5cbe0" }}>{title}</h2>
      {children}
    </section>
  );
}

export function SampleBanner() {
  return (
    <div
      style={{
        background: "#2a2410",
        border: "1px solid #5a4a1a",
        color: "#e8d48a",
        padding: "10px 14px",
        borderRadius: 8,
        fontSize: 13,
        marginBottom: 20,
      }}
    >
      Showing <b>sample data</b> — run the pipeline (<code>./run_pipeline.sh</code>) to populate
      <code> ../data/scores.json</code> with a real lecture.
    </div>
  );
}

export function QualityPill({ label, count }: { label: string; count: number }) {
  return (
    <span
      style={{
        display: "inline-flex",
        gap: 6,
        alignItems: "center",
        background: "#1d2230",
        borderRadius: 20,
        padding: "3px 10px",
        fontSize: 12,
        marginRight: 6,
      }}
    >
      <span style={{ width: 8, height: 8, borderRadius: 8, background: QUALITY_COLORS[label] ?? "#888" }} />
      {label} · {count}
    </span>
  );
}
