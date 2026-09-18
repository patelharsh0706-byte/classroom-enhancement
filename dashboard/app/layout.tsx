import Link from "next/link";
import type { ReactNode } from "react";

export const metadata = {
  title: "Class Engagement Dashboard",
  description: "Lecture participation — scored and gamified",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          fontFamily: "ui-sans-serif, system-ui, -apple-system, sans-serif",
          background: "#0f1117",
          color: "#e6e8ee",
        }}
      >
        <nav
          style={{
            display: "flex",
            gap: 20,
            padding: "16px 28px",
            borderBottom: "1px solid #232838",
            alignItems: "center",
          }}
        >
          <span style={{ fontWeight: 700, letterSpacing: 0.3 }}>🎓 Class Engagement</span>
          <Link href="/professor" style={navLink}>Professor</Link>
          <Link href="/students" style={navLink}>Students</Link>
        </nav>
        <main style={{ padding: "28px", maxWidth: 980, margin: "0 auto" }}>{children}</main>
      </body>
    </html>
  );
}

const navLink = { color: "#9bb0ff", textDecoration: "none", fontSize: 15 };
