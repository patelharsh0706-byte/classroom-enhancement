"use client";

export default function AudioPlayer({ filename }: { filename: string }) {
  const src = `/api/audio?file=${encodeURIComponent(filename)}`;
  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ fontSize: 13, color: "#8a90a2", marginBottom: 6 }}>
        Lecture recording — <span style={{ color: "#b9c0d4" }}>{filename}</span>
      </div>
      <audio
        controls
        src={src}
        style={{
          width: "100%",
          accentColor: "#6c8fff",
          borderRadius: 8,
          outline: "none",
        }}
      />
    </div>
  );
}
