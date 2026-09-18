"use client";

import { useState } from "react";
import type { Candidate, TimelineTurn } from "@/lib/data";

/**
 * Inline correction control, shown only on turns whose cluster needs review.
 *
 * Two deliberate choices, both load-bearing:
 *
 * 1. The system's guess is NOT pre-selected. A pre-filled default turns every
 *    confirmation into a click-through, and a store full of reflexive
 *    confirmations is worse than an empty one — the threshold calibration that
 *    reads it later would treat them as evidence.
 *
 * 2. There is a play button. Three seconds of listening is the difference
 *    between evidence and a rubber stamp.
 */
export default function SpeakerPicker({
  turn,
  roster,
  clusterTurnCount,
  clusterSpan,
  onSeek,
  onSubmit,
  onClose,
}: {
  turn: TimelineTurn;
  roster: string[];
  clusterTurnCount: number;
  clusterSpan: string;
  onSeek: (s: number) => void;
  onSubmit: (v: {
    verdict: "confirm" | "correct" | "unenrolled" | "unknown";
    name: string;
    consentToEnroll: boolean;
  }) => Promise<void>;
  onClose: () => void;
}) {
  const [mode, setMode] = useState<"pick" | "roster" | "new">("pick");
  const [newName, setNewName] = useState("");
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const cands = turn.candidates ?? [];
  const systemGuess = cands[0]?.name;

  async function send(
    verdict: "confirm" | "correct" | "unenrolled" | "unknown",
    name: string,
    consentToEnroll = false,
  ) {
    setBusy(true);
    setErr(null);
    try {
      await onSubmit({ verdict, name, consentToEnroll });
      onClose();
    } catch (e) {
      setErr((e as Error).message);
      setBusy(false);
    }
  }

  return (
    <div
      onClick={(e) => e.stopPropagation()}
      style={{
        marginTop: 8, padding: 12, background: "#0d1017",
        border: "1px solid #2b3346", borderRadius: 10, fontSize: 12,
      }}
    >
      <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 4 }}>
        <b style={{ color: "#e8ecf7", fontSize: 12 }}>Who is this?</b>
        <span style={{ color: "#8a90a2" }}>{reasonText(turn)}</span>
        <button onClick={onClose} style={linkBtn}>close</button>
      </div>
      <div style={{ color: "#7d8497", marginBottom: 10 }}>
        Relabels all {clusterTurnCount} turn{clusterTurnCount === 1 ? "" : "s"} by this speaker
        {clusterSpan ? ` (${clusterSpan})` : ""}.
        <button onClick={() => onSeek(turn.start)} style={{ ...linkBtn, marginLeft: 8 }}>
          ▶ listen
        </button>
      </div>

      {mode === "pick" && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          {cands.map((c) => (
            <CandidateButton
              key={c.name}
              c={c}
              disabled={busy}
              onClick={() => send(c.name === systemGuess ? "confirm" : "correct", c.name)}
            />
          ))}
          <button style={ghostBtn} disabled={busy} onClick={() => setMode("roster")}>
            Someone else…
          </button>
          <button style={ghostBtn} disabled={busy} onClick={() => setMode("new")}>
            Not enrolled / new person
          </button>
          <button style={ghostBtn} disabled={busy} onClick={() => send("unknown", "")}>
            Can&apos;t tell
          </button>
        </div>
      )}

      {mode === "roster" && (
        <div>
          <div style={{ maxHeight: 150, overflowY: "auto", display: "flex", flexWrap: "wrap", gap: 5 }}>
            {roster.map((n) => (
              <button key={n} style={ghostBtn} disabled={busy} onClick={() => send("correct", n)}>
                {n}
              </button>
            ))}
          </div>
          <button style={{ ...linkBtn, marginTop: 8 }} onClick={() => setMode("pick")}>← back</button>
        </div>
      )}

      {mode === "new" && (
        <div>
          <input
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="Full name"
            style={{
              width: "100%", padding: "6px 8px", background: "#161a24", color: "#e8ecf7",
              border: "1px solid #2b3346", borderRadius: 6, fontSize: 12,
            }}
          />
          <label style={{ display: "block", margin: "8px 0", color: "#8a90a2", cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={consent}
              onChange={(e) => setConsent(e.target.checked)}
              style={{ marginRight: 6 }}
            />
            Also build a voiceprint from this lecture audio —{" "}
            <b style={{ color: "#d16a8a" }}>only if they consented to voice recording</b>
          </label>
          <div style={{ display: "flex", gap: 6 }}>
            <button
              style={primaryBtn}
              disabled={busy || !newName.trim()}
              onClick={() => send("unenrolled", newName.trim(), consent)}
            >
              Save
            </button>
            <button style={ghostBtn} onClick={() => setMode("pick")}>← back</button>
          </div>
        </div>
      )}

      {err && <div style={{ color: "#ff8a8a", marginTop: 8 }}>{err}</div>}
    </div>
  );
}

function CandidateButton({ c, disabled, onClick }: { c: Candidate; disabled: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        ...ghostBtn,
        borderColor: "#3a4459",
        display: "inline-flex", gap: 6, alignItems: "baseline",
      }}
    >
      <b style={{ color: "#e8ecf7" }}>{c.name}</b>
      <span style={{ color: "#8a90a2", fontSize: 11 }}>
        {c.sim.toFixed(3)}
        {c.z !== null && c.z !== undefined ? ` · z ${c.z.toFixed(1)}` : ""}
      </span>
    </button>
  );
}

function reasonText(t: TimelineTurn): string {
  switch (t.review?.reason) {
    case "unknown":
      return "score below the review floor — may not be enrolled at all";
    case "low-sim":
      return "low confidence";
    case "confusable":
      return `⚠ top two sound alike (${t.review.confusableWith})`;
    case "audit":
      return "spot check";
    default:
      return "";
  }
}

const ghostBtn: React.CSSProperties = {
  background: "#161a24", color: "#b9c0d4", border: "1px solid #2b3346",
  borderRadius: 6, padding: "5px 9px", fontSize: 12, cursor: "pointer",
};
const primaryBtn: React.CSSProperties = {
  ...ghostBtn, background: "#2b4a7a", color: "#fff", borderColor: "#3a5f96",
};
const linkBtn: React.CSSProperties = {
  background: "none", border: "none", color: "#6c8fff", cursor: "pointer",
  fontSize: 11, padding: 0, textDecoration: "underline",
};
