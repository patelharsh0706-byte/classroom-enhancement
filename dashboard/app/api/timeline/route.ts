import { getTimeline, getSourceAudio } from "@/lib/data";

// The pipeline rewrites ../data/*.json while a class is being processed, so this
// must never be cached — the point of the endpoint is that the page picks up
// new turns as they land.
export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET() {
  const { turns, profLabel } = getTimeline();
  return Response.json(
    { turns, profLabel, filename: getSourceAudio() },
    { headers: { "Cache-Control": "no-store" } },
  );
}
