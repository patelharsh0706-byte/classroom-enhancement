"use client";

import { useState } from "react";

/**
 * Learning is a deliberate, explicit act — never a side effect of reviewing.
 *
 * Corrections are stored the instant they're made, but voiceprints only change
 * when someone presses this. That keeps review fast, and it means a misclick
 * during review cannot silently contaminate a reference vector. Dry run first
 * is the default habit this UI tries to encourage.
 */
export default function LearnButton({ pendingCount }: { pendingCount: number }) {
  const [busy, setBusy] = useState<"" | "dry" | "real">("");
  const [out, setOut] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function run(dry: boolean) {
    setBusy(dry ? "dry" : "real");
    setErr(null);
    setOut(null);
    try {
      const r = await fetch("/api/learn", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dry_run: dry }),
      });
      const d = await r.json();
      if (!r.ok || !d.ok) setErr(d.error ?? `failed (${r.status})`);
      setOut(d.output || null);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy("");
    }
  }

  return (
    <div
      style={{
        background: "#12211a", border: "1px solid #1f4535", borderRadius: 10,
        padding: 14, marginBottom: 20,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <div style={{ flex: 1, minWidth: 260 }}>
          <b style={{ color: "#9fe0bd", fontSize: 14 }}>Learn from verified speakers</b>
          <div style={{ color: "#7d9a8b", fontSize: 12, marginTop: 3, lineHeight: 1.5 }}>
            {pendingCount} confirmed cluster{pendingCount === 1 ? "" : "s"} on file. Confirmed
            audio was recorded through the <i>room mic</i> — the channel the phone enrollment
            clips lack — so folding it in closes the gap that makes matching hard.
          </div>
        </div>
        <button onClick={() => run(true)} disabled={!!busy} style={ghost}>
          {busy === "dry" ? "checking…" : "Dry run"}
        </button>
        <button onClick={() => run(false)} disabled={!!busy} style={primary}>
          {busy === "real" ? "learning…" : "Apply learning"}
        </button>
      </div>

      {err && (
        <pre style={{ ...pre, color: "#ff9a9a", borderColor: "#5a2a2a" }}>{err}</pre>
      )}
      {out && <pre style={pre}>{out}</pre>}
    </div>
  );
}

const ghost: React.CSSProperties = {
  background: "#161a24", color: "#b9c0d4", border: "1px solid #2b3346",
  borderRadius: 6, padding: "7px 12px", fontSize: 13, cursor: "pointer",
};
const primary: React.CSSProperties = {
  ...ghost, background: "#1f4535", color: "#9fe0bd", borderColor: "#2f6b50",
};
const pre: React.CSSProperties = {
  marginTop: 12, marginBottom: 0, padding: 12, background: "#0d1017",
  border: "1px solid #232838", borderRadius: 8, fontSize: 11.5,
  color: "#b9c0d4", overflowX: "auto", whiteSpace: "pre", lineHeight: 1.5,
  maxHeight: 380, overflowY: "auto",
};
