import path from "path";
import { execFile } from "child_process";
import { promisify } from "util";
import { NextRequest } from "next/server";

export const dynamic = "force-dynamic";
export const revalidate = 0;
export const maxDuration = 600;

const execFileAsync = promisify(execFile);
const ROOT = path.join(process.cwd(), "..");

/**
 * Thin shell around scripts/apply_learning.py. All the logic — the safety gates,
 * the ledger, the retraction path — lives in Python so it runs identically from
 * a terminal, where it can be inspected before being trusted.
 */
export async function POST(req: NextRequest) {
  let dryRun = false;
  try {
    dryRun = (await req.json())?.dry_run === true;
  } catch {
    /* no body is fine */
  }

  const python = path.join(ROOT, ".venv", "bin", "python");
  const args = [path.join(ROOT, "scripts", "apply_learning.py")];
  if (dryRun) args.push("--dry-run");

  try {
    const { stdout, stderr } = await execFileAsync(python, args, {
      cwd: ROOT,
      maxBuffer: 8 * 1024 * 1024,
      timeout: 9 * 60 * 1000,
    });
    // The interesting output is the ingest log and the before/after table;
    // model-loading chatter on stderr is noise unless it actually failed.
    return Response.json({ ok: true, dryRun, output: stdout, warnings: stderr.slice(-2000) });
  } catch (e) {
    const err = e as Error & { stdout?: string; stderr?: string };
    return Response.json(
      { ok: false, error: err.message, output: err.stdout ?? "", warnings: (err.stderr ?? "").slice(-4000) },
      { status: 500 },
    );
  }
}
