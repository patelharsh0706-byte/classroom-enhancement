"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

export type ExpTurn = {
  start: number;      // relative to this window's audio file — drives sync
  end: number;
  absLabel: string;   // absolute lecture time — for display
  name: string;
  isProf: boolean;
  sim: number;
  status: "confirmed" | "unenrolled" | "unverified";
  systemGuess: string;
  text: string;
};

export type ExpSpeaker = {
  name: string; status: string; sim: number;
  turns: number; seconds: number; systemGuess: string;
};

export type ExpWindow = {
  id: string; label: string; note: string; audio: string;
  duration: number; turns: ExpTurn[]; speakers: ExpSpeaker[];
};

const PROF_HUE = "#e0a35e";
const STUDENT_HUE = "#6c8fff";
const UNENROLLED_HUE = "#d16a8a";

const hueFor = (t: { isProf: boolean; status: string }) =>
  t.isProf ? PROF_HUE : t.status === "unenrolled" ? UNENROLLED_HUE : STUDENT_HUE;

export default function ExperimentWindow({ win }: { win: ExpWindow }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [now, setNow] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [follow, setFollow] = useState(true);

  // timeupdate fires ~4x/sec — too coarse for word-level highlighting.
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

  const students = useMemo(() => win.turns.filter((t) => !t.isProf), [win.turns]);
  const profs = useMemo(() => win.turns.filter((t) => t.isProf), [win.turns]);
  const activeStart = useMemo(() => {
    const hit = win.turns.find((t) => now >= t.start && now < t.end);
    return hit ? hit.start : null;
  }, [win.turns, now]);

  const seek = useCallback((s: number) => {
    const a = audioRef.current;
    if (!a) return;
    a.currentTime = s;
    setNow(s);
    void a.play();
  }, []);

  const barRef = useRef<HTMLDivElement>(null);

  return (
    <section
      style={{
        background: "#161a24", border: "1px solid #232838",
        borderRadius: 12, padding: 20, marginBottom: 24,
      }}
    >
      <h2 style={{ margin: "0 0 4px", fontSize: 17, color: "#e8ecf7" }}>{win.label}</h2>
      <p style={{ margin: "0 0 14px", fontSize: 13, color: "#8a90a2", lineHeight: 1.5 }}>{win.note}</p>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 14 }}>
        {win.speakers.map((s) => (
          <SpeakerChip key={s.name} s={s} />
        ))}
      </div>

      <audio
        ref={audioRef}
        controls
        src={`/api/audio?file=${encodeURIComponent(win.audio)}`}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onTimeUpdate={(e) => !playing && setNow(e.currentTarget.currentTime)}
        onSeeked={(e) => setNow(e.currentTarget.currentTime)}
        style={{ width: "100%", accentColor: STUDENT_HUE, borderRadius: 8, outline: "none" }}
      />

      {/* Ribbon: professor on the bottom half, students on the top. Click to jump. */}
      <div
        ref={barRef}
        onClick={(e) => {
          const r = barRef.current?.getBoundingClientRect();
          if (r) seek(((e.clientX - r.left) / r.width) * win.duration);
        }}
        title="Click to jump"
        style={{
          position: "relative", height: 34, marginTop: 14, background: "#10131c",
          border: "1px solid #232838", borderRadius: 8, overflow: "hidden", cursor: "pointer",
        }}
      >
        {win.turns.map((t) => (
          <div
            key={`${t.name}@${t.start}`}
            style={{
              position: "absolute",
              left: `${(100 * t.start) / win.duration}%`,
              width: `${Math.max(0.3, (100 * (t.end - t.start)) / win.duration)}%`,
              top: t.isProf ? 17 : 0,
              height: 17,
              background: hueFor(t),
              opacity: t.start === activeStart ? 1 : 0.45,
            }}
          />
        ))}
        <div
          style={{
            position: "absolute", left: `${(100 * now) / win.duration}%`,
            top: 0, bottom: 0, width: 2, background: "#fff",
            boxShadow: "0 0 6px rgba(255,255,255,.8)",
          }}
        />
      </div>

      <label
        style={{
          display: "block", textAlign: "right", fontSize: 12, color: "#8a90a2",
          margin: "10px 0", cursor: "pointer", userSelect: "none",
        }}
      >
        <input
          type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)}
          style={{ marginRight: 6, accentColor: STUDENT_HUE }}
        />
        auto-scroll
      </label>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, alignItems: "start" }}>
        <Lane title="Students" accent={STUDENT_HUE} turns={students} now={now} follow={follow} onSeek={seek} />
        <Lane title="Professor" accent={PROF_HUE} turns={profs} now={now} follow={follow} onSeek={seek} />
      </div>
    </section>
  );
}

function SpeakerChip({ s }: { s: ExpSpeaker }) {
  const unenrolled = s.status === "unenrolled";
  const color = s.name === "Prof" ? PROF_HUE : unenrolled ? UNENROLLED_HUE : STUDENT_HUE;
  return (
    <span
      style={{
        display: "inline-flex", alignItems: "center", gap: 7, background: "#1d2230",
        border: `1px solid ${color}44`, borderRadius: 20, padding: "5px 12px", fontSize: 12,
      }}
      title={
        unenrolled
          ? `No voiceprint. System scored its nearest guess (${s.systemGuess}) at ${s.sim.toFixed(3)} and correctly declined to name them.`
          : `Matched at ${s.sim.toFixed(3)}, confirmed by human review.`
      }
    >
      <span style={{ width: 8, height: 8, borderRadius: 8, background: color }} />
      <b style={{ color: "#e8ecf7", fontWeight: 600 }}>{s.name}</b>
      <span style={{ color: "#8a90a2" }}>
        {s.seconds}s · {s.turns} turns
      </span>
      {unenrolled ? (
        <span style={{ color: UNENROLLED_HUE, fontSize: 11 }}>⚠ not enrolled</span>
      ) : (
        <span style={{ color: "#5ee0a0", fontSize: 11 }}>✓ {s.sim.toFixed(2)}</span>
      )}
    </span>
  );
}

function Lane({
  title, accent, turns, now, follow, onSeek,
}: {
  title: string; accent: string; turns: ExpTurn[];
  now: number; follow: boolean; onSeek: (s: number) => void;
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
          padding: "8px 12px", borderBottom: "1px solid #232838",
          fontSize: 13, fontWeight: 600, color: accent,
        }}
      >
        {title}
      </div>
      <div ref={scrollRef} style={{ maxHeight: 420, overflowY: "auto", padding: 8 }}>
        {turns.length === 0 && (
          <p style={{ color: "#5a6072", fontSize: 13, padding: 8, margin: 0 }}>No turns.</p>
        )}
        {turns.map((t) => {
          const active = now >= t.start && now < t.end;
          const past = now >= t.end;
          const hue = hueFor(t);
          return (
            <div
              key={`${t.name}@${t.start}`}
              ref={active ? activeRef : undefined}
              onClick={() => onSeek(t.start)}
              style={{
                padding: "8px 10px", marginBottom: 6, borderRadius: 8, cursor: "pointer",
                background: active ? "#1c2434" : "transparent",
                borderLeft: `3px solid ${active ? hue : past ? "#2b3142" : "transparent"}`,
              }}
            >
              <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 3 }}>
                <span style={{ fontSize: 11, color: "#5a6072", fontVariantNumeric: "tabular-nums" }}>
                  {t.absLabel}
                </span>
                <span style={{ fontSize: 12, fontWeight: 600, color: active ? "#fff" : "#b9c0d4" }}>
                  {t.name}
                </span>
                {t.status === "unenrolled" ? (
                  <span style={{ fontSize: 10, color: UNENROLLED_HUE }}>⚠ no voiceprint</span>
                ) : (
                  <span style={{ fontSize: 10, color: "#5ee0a0" }}>✓ {t.sim.toFixed(2)}</span>
                )}
              </div>
              <Words text={t.text} start={t.start} end={t.end} now={now} active={active} past={past} />
            </div>
          );
        })}
      </div>
    </div>
  );
}

/**
 * Karaoke highlighting. Only turn-level timing exists, so word position is linear
 * interpolation across the turn — exact at the boundaries, approximate inside.
 */
function Words({
  text, start, end, now, active, past,
}: {
  text: string; start: number; end: number; now: number; active: boolean; past: boolean;
}) {
  const words = useMemo(() => text.split(/\s+/).filter(Boolean), [text]);
  if (!active) {
    return <div style={{ fontSize: 13, lineHeight: 1.55, color: past ? "#7d8497" : "#5a6072" }}>{text}</div>;
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
