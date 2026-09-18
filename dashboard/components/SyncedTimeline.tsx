"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { TimelineTurn } from "@/lib/data";
import { QUALITY_COLORS } from "@/components/ui";
import SpeakerPicker from "@/components/SpeakerPicker";

const STUDENT_HUE = "#6c8fff";
const PROF_HUE = "#e0a35e";
const POLL_MS = 5000;

/** Cheap content fingerprint — catches relabels, which never change turn count. */
const signature = (ts: TimelineTurn[]) =>
  ts.map((t) => `${t.clusterId ?? ""}:${t.name}:${t.start}`).join("|");

export default function SyncedTimeline({
  initialTurns,
  initialFilename,
  roster = [],
  profLabel = "Prof",
}: {
  initialTurns: TimelineTurn[];
  initialFilename: string | null;
  roster?: string[];
  profLabel?: string;
}) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [rawTurns, setTurns] = useState(initialTurns);
  const [filename, setFilename] = useState(initialFilename);
  const [now, setNow] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [follow, setFollow] = useState(true);
  const [openPicker, setOpenPicker] = useState<string | null>(null);
  // Optimistic relabels, keyed by CLUSTER — correcting one turn corrects the
  // speaker, so every turn in that cluster moves together.
  const [overrides, setOverrides] = useState<Record<string, string>>({});

  const turns = useMemo(
    () =>
      rawTurns.map((t) => {
        const o = t.clusterId ? overrides[t.clusterId] : undefined;
        if (!o) return t;
        // isProf recomputed so a relabel moves the row between lanes.
        return { ...t, name: o, isProf: o === profLabel, review: { ...t.review!, needed: false } };
      }),
    [rawTurns, overrides, profLabel],
  );

  const clusterInfo = useMemo(() => {
    const m: Record<string, { count: number; start: number; end: number }> = {};
    for (const t of turns) {
      if (!t.clusterId) continue;
      const e = (m[t.clusterId] ??= { count: 0, start: t.start, end: t.end });
      e.count += 1;
      e.start = Math.min(e.start, t.start);
      e.end = Math.max(e.end, t.end);
    }
    return m;
  }, [turns]);

  const submitCorrection = useCallback(
    async (turn: TimelineTurn, v: { verdict: string; name: string; consentToEnroll: boolean }) => {
      const cid = turn.clusterId;
      if (!cid) throw new Error("this turn has no cluster id");
      const applied = v.verdict === "unknown" ? turn.name : v.name;
      setOverrides((p) => ({ ...p, [cid]: applied }));
      const res = await fetch("/api/corrections", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cluster_id: cid,
          verdict: v.verdict,
          corrected_name: v.name,
          candidates: turn.candidates,
          system_guess: turn.candidates?.[0]?.name,
          system_sim: turn.candidates?.[0]?.sim,
          system_z: turn.candidates?.[0]?.z,
          review_reason: turn.review?.reason,
          consent_to_enroll: v.consentToEnroll,
        }),
      });
      if (!res.ok) {
        setOverrides((p) => {
          const n = { ...p };
          delete n[cid]; // roll back so the UI never claims a save that failed
          return n;
        });
        throw new Error((await res.json().catch(() => ({}))).error ?? `save failed (${res.status})`);
      }
    },
    [],
  );

  // Poll for new turns so a class still being processed fills in live, rather
  // than requiring a reload once the pipeline finishes writing.
  useEffect(() => {
    const id = setInterval(async () => {
      try {
        const r = await fetch("/api/timeline", { cache: "no-store" });
        if (!r.ok) return;
        const d = (await r.json()) as { turns: TimelineTurn[]; filename: string | null };
        // Compare content, not length: a correction relabels turns without
        // changing how many there are, so a length check would discard exactly
        // the update this poll exists to deliver.
        setTurns((prev) => (signature(d.turns) === signature(prev) ? prev : d.turns));
        setFilename((prev) => d.filename ?? prev);
      } catch {
        /* transient dev-server hiccup — keep the last good data */
      }
    }, POLL_MS);
    return () => clearInterval(id);
  }, []);

  // <audio> only fires timeupdate ~4x/sec, which is too coarse for word-level
  // highlighting. Drive from rAF while playing, and fall back to the event when
  // paused (so scrubbing still moves the highlight).
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const tick = () => {
      const a = audioRef.current;
      if (a) setNow(a.currentTime);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  const students = useMemo(() => turns.filter((t) => !t.isProf), [turns]);
  const profs = useMemo(() => turns.filter((t) => t.isProf), [turns]);

  const activeStart = useMemo(() => {
    const hit = turns.find((t) => now >= t.start && now < t.end);
    return hit ? hit.start : null;
  }, [turns, now]);

  const seek = useCallback((s: number) => {
    const a = audioRef.current;
    if (!a) return;
    a.currentTime = s;
    setNow(s);
    void a.play();
  }, []);

  const span = duration || (turns.length ? turns[turns.length - 1].end : 0);

  if (!filename && turns.length === 0) {
    return (
      <p style={{ color: "#8a90a2", margin: 0 }}>
        No transcript yet. Run the pipeline to populate <code>../data/named_turns.json</code>.
      </p>
    );
  }

  return (
    <div>
      {filename && (
        <>
          <div style={{ fontSize: 13, color: "#8a90a2", marginBottom: 6 }}>
            Lecture recording — <span style={{ color: "#b9c0d4" }}>{filename}</span>
          </div>
          <audio
            ref={audioRef}
            controls
            src={`/api/audio?file=${encodeURIComponent(filename)}`}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onTimeUpdate={(e) => !playing && setNow(e.currentTarget.currentTime)}
            onSeeked={(e) => setNow(e.currentTarget.currentTime)}
            onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
            style={{ width: "100%", accentColor: STUDENT_HUE, borderRadius: 8, outline: "none" }}
          />
        </>
      )}

      {/* Whole-lecture ribbon: who held the floor, when. Click to jump. */}
      <Ribbon turns={turns} span={span} now={now} onSeek={seek} activeStart={activeStart} />

      <div style={{ display: "flex", gap: 10, alignItems: "center", margin: "14px 0 10px" }}>
        <Legend color={STUDENT_HUE} label={`Students · ${students.length}`} />
        <Legend color={PROF_HUE} label={`Professor · ${profs.length}`} />
        <label
          style={{ marginLeft: "auto", fontSize: 12, color: "#8a90a2", cursor: "pointer", userSelect: "none" }}
        >
          <input
            type="checkbox"
            checked={follow}
            onChange={(e) => setFollow(e.target.checked)}
            style={{ marginRight: 6, accentColor: STUDENT_HUE }}
          />
          auto-scroll
        </label>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, alignItems: "start" }}>
        <Lane
          title="Students"
          accent={STUDENT_HUE}
          turns={students}
          now={now}
          follow={follow}
          onSeek={seek}
          emptyHint="No student turns matched. Every student turn is currently Unknown."
          roster={roster}
          clusterInfo={clusterInfo}
          openPicker={openPicker}
          setOpenPicker={setOpenPicker}
          onCorrect={submitCorrection}
        />
        <Lane
          title="Professor"
          accent={PROF_HUE}
          turns={profs}
          now={now}
          follow={follow}
          onSeek={seek}
          emptyHint="No professor turns."
          roster={roster}
          clusterInfo={clusterInfo}
          openPicker={openPicker}
          setOpenPicker={setOpenPicker}
          onCorrect={submitCorrection}
        />
      </div>

      <p style={{ fontSize: 11, color: "#5a6072", marginTop: 12, lineHeight: 1.5 }}>
        Word highlighting is <b>interpolated</b> across each turn&apos;s duration — the pipeline stores
        start/end per turn, not per word. Turn boundaries are exact; the position within a long turn is an
        estimate.
      </p>
    </div>
  );
}

function Ribbon({
  turns,
  span,
  now,
  onSeek,
  activeStart,
}: {
  turns: TimelineTurn[];
  span: number;
  now: number;
  onSeek: (s: number) => void;
  activeStart: number | null;
}) {
  const barRef = useRef<HTMLDivElement>(null);
  if (!span) return null;

  return (
    <div
      ref={barRef}
      onClick={(e) => {
        const r = barRef.current?.getBoundingClientRect();
        if (r) onSeek(((e.clientX - r.left) / r.width) * span);
      }}
      title="Click to jump"
      style={{
        position: "relative",
        height: 34,
        marginTop: 14,
        background: "#10131c",
        border: "1px solid #232838",
        borderRadius: 8,
        overflow: "hidden",
        cursor: "pointer",
      }}
    >
      {turns.map((t) => (
        <div
          key={`${t.clusterId ?? t.name}@${t.start}`}
          style={{
            position: "absolute",
            left: `${(100 * t.start) / span}%`,
            width: `${Math.max(0.15, (100 * (t.end - t.start)) / span)}%`,
            top: t.isProf ? 17 : 0,
            height: 17,
            background: t.isProf ? PROF_HUE : STUDENT_HUE,
            opacity: t.start === activeStart ? 1 : 0.45,
          }}
        />
      ))}
      <div
        style={{
          position: "absolute",
          left: `${(100 * now) / span}%`,
          top: 0,
          bottom: 0,
          width: 2,
          background: "#fff",
          boxShadow: "0 0 6px rgba(255,255,255,.8)",
        }}
      />
    </div>
  );
}

function Lane({
  title,
  accent,
  turns,
  now,
  follow,
  onSeek,
  emptyHint,
  roster,
  clusterInfo,
  openPicker,
  setOpenPicker,
  onCorrect,
}: {
  title: string;
  accent: string;
  turns: TimelineTurn[];
  now: number;
  follow: boolean;
  onSeek: (s: number) => void;
  emptyHint: string;
  roster: string[];
  clusterInfo: Record<string, { count: number; start: number; end: number }>;
  openPicker: string | null;
  setOpenPicker: (k: string | null) => void;
  onCorrect: (
    t: TimelineTurn,
    v: { verdict: string; name: string; consentToEnroll: boolean },
  ) => Promise<void>;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const activeRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!follow || !activeRef.current || !scrollRef.current) return;
    const box = scrollRef.current.getBoundingClientRect();
    const row = activeRef.current.getBoundingClientRect();
    if (row.top < box.top || row.bottom > box.bottom) {
      activeRef.current.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, [now, follow]);

  return (
    <div style={{ background: "#10131c", border: "1px solid #232838", borderRadius: 10 }}>
      <div
        style={{
          padding: "8px 12px",
          borderBottom: "1px solid #232838",
          fontSize: 13,
          fontWeight: 600,
          color: accent,
          position: "sticky",
          top: 0,
        }}
      >
        {title}
      </div>
      <div ref={scrollRef} style={{ maxHeight: 460, overflowY: "auto", padding: 8 }}>
        {turns.length === 0 && (
          <p style={{ color: "#5a6072", fontSize: 13, padding: 8, margin: 0 }}>{emptyHint}</p>
        )}
        {turns.map((t) => {
          const active = now >= t.start && now < t.end;
          const past = now >= t.end;
          const pickerKey = `${t.clusterId ?? t.name}@${t.start}`;
          const info = t.clusterId ? clusterInfo[t.clusterId] : undefined;
          return (
            <div
              key={`${t.clusterId ?? t.name}@${t.start}`}
              ref={active ? activeRef : undefined}
              onClick={() => onSeek(t.start)}
              style={{
                padding: "8px 10px",
                marginBottom: 6,
                borderRadius: 8,
                cursor: "pointer",
                background: active ? "#1c2434" : "transparent",
                borderLeft: `3px solid ${active ? accent : past ? "#2b3142" : "transparent"}`,
                transition: "background .15s",
              }}
            >
              <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 3 }}>
                <span
                  title="Time in the original lecture recording"
                  style={{ fontSize: 11, color: "#5a6072", fontVariantNumeric: "tabular-nums" }}
                >
                  {t.absLabel ?? fmt(t.start)}
                </span>
                <span
                  style={{
                    fontSize: 12, fontWeight: 600,
                    color: t.name === "Unknown" ? "#8a90a2" : active ? "#fff" : "#b9c0d4",
                    fontStyle: t.name === "Unknown" ? "italic" : "normal",
                  }}
                >
                  {t.name}
                </span>
                {t.human && <span style={{ fontSize: 10, color: "#5ee0a0" }}>✓ verified</span>}
                {/* The three candidates sit inline, one click each — the point is
                    that resolving a speaker costs a single click, not a click to
                    open something and then another to choose. */}
                {t.review?.needed &&
                  (t.candidates ?? []).slice(0, 3).map((c) => (
                    <button
                      key={c.name}
                      onClick={(e) => {
                        e.stopPropagation();
                        void onCorrect(t, {
                          verdict: c.name === t.candidates?.[0]?.name ? "confirm" : "correct",
                          name: c.name,
                          consentToEnroll: false,
                        });
                      }}
                      title={`Assign ${c.name} to all turns by this speaker (similarity ${c.sim.toFixed(3)})`}
                      style={{
                        fontSize: 10.5, color: "#cfe0ff", background: "#1b2436",
                        border: "1px solid #32405a", borderRadius: 10,
                        padding: "1px 8px", cursor: "pointer",
                      }}
                    >
                      {c.name} <span style={{ color: "#7d8497" }}>{c.sim.toFixed(2)}</span>
                    </button>
                  ))}
                {t.review?.needed && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setOpenPicker(pickerKey === openPicker ? null : pickerKey);
                    }}
                    title="Someone else, a person not enrolled, or listen first"
                    style={{
                      fontSize: 10, color: "#ffcf7a", background: "#2a2410",
                      border: "1px solid #5a4a1a", borderRadius: 10,
                      padding: "1px 7px", cursor: "pointer",
                    }}
                  >
                    {t.review.reason === "confusable" ? "⚠ other…" : "other…"}
                  </button>
                )}
                {t.quality && (
                  <span style={{ fontSize: 11, color: QUALITY_COLORS[t.quality] ?? "#888" }}>
                    ● {t.quality}
                    {typeof t.points === "number" ? ` +${t.points}` : ""}
                  </span>
                )}
              </div>
              <Words text={t.text} start={t.start} end={t.end} now={now} active={active} past={past} />
              {openPicker === pickerKey && (
                <SpeakerPicker
                  turn={t}
                  roster={roster}
                  clusterTurnCount={info?.count ?? 1}
                  clusterSpan={info ? `${fmt(info.start)}–${fmt(info.end)}` : ""}
                  onSeek={onSeek}
                  onSubmit={(v) => onCorrect(t, v)}
                  onClose={() => setOpenPicker(null)}
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/**
 * Karaoke highlighting. We only have turn-level timing, so word position is
 * linear interpolation across the turn — exact at the boundaries, approximate
 * in between. Deliberately not presented as word-accurate timing.
 */
function Words({
  text,
  start,
  end,
  now,
  active,
  past,
}: {
  text: string;
  start: number;
  end: number;
  now: number;
  active: boolean;
  past: boolean;
}) {
  const words = useMemo(() => text.split(/\s+/).filter(Boolean), [text]);

  if (!active) {
    return (
      <div style={{ fontSize: 13, lineHeight: 1.55, color: past ? "#7d8497" : "#5a6072" }}>{text}</div>
    );
  }

  const p = Math.min(1, Math.max(0, (now - start) / Math.max(0.001, end - start)));
  const cursor = Math.floor(p * words.length);

  return (
    <div style={{ fontSize: 13, lineHeight: 1.55 }}>
      {words.map((w, i) => (
        <span
          key={i}
          style={{
            color: i < cursor ? "#e8ecf7" : i === cursor ? "#fff" : "#6b7285",
            background: i === cursor ? "rgba(108,143,255,.35)" : "transparent",
            borderRadius: 3,
            padding: i === cursor ? "0 2px" : 0,
          }}
        >
          {w}{" "}
        </span>
      ))}
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span style={{ display: "inline-flex", gap: 6, alignItems: "center", fontSize: 12, color: "#8a90a2" }}>
      <span style={{ width: 10, height: 10, borderRadius: 3, background: color }} />
      {label}
    </span>
  );
}

function fmt(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}
