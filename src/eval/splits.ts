/**
 * npm run eval:splits
 *
 * Every candidate touch split on the synthetic and tuning clips with its
 * evidence (kink score, velocity jump, players in reach), whether it was kept
 * and whether a true touch is there. For choosing the keep/drop rule from
 * data. Holdout clips are left out.
 */
import { allCases, HOLDOUT } from "./dataset";
import { runFixture } from "./pipeline";
const rows: string[] = [];
for (const c of allCases()) {
  if (!c.truth || HOLDOUT.includes(c.name) || c.knownLimit) continue;
  const fps = c.fixture.video.fps;
  const r = runFixture(c.fixture, undefined, false, c.truth.court)!;
  for (const s of r.result.touchSplits ?? []) {
    const mid = (s.frameBefore + s.frameAfter) / 2;
    const near = c.truth.touches.map((t) => Math.abs(t.frame - mid) / fps);
    const d = Math.min(...near);
    const real = d <= 0.15 || c.truth.touches.some((t) => t.frame >= s.frameBefore - 1 && t.frame <= s.frameAfter + 1);
    rows.push(`${c.synthetic ? "S" : "R"} ${c.name.padEnd(16)} f${s.frameBefore}-${s.frameAfter} gap ${s.frameAfter - s.frameBefore} kink ${s.kink == null ? "  gap" : s.kink.toFixed(0).padStart(5)} dv ${Number.isFinite(s.dvMps) ? s.dvMps.toFixed(1).padStart(5) : "  inf"} cands ${s.candidates} ${s.kept ? "kept" : "drop"} ${real ? "TOUCH" : "-"}`);
  }
}
console.log(rows.join("\n"));
