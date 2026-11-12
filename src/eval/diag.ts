/**
 * npm run eval:diag -- <case>[,<case>...] [--log]
 *
 * Where one clip goes wrong, as text: each true touch against the found one,
 * the path error flight by flight (synthetic), the solver's own account of
 * every segment, and the found touches nothing was labelled at.
 */
import { allCases } from "./dataset";
import { evaluateCase } from "./evaluate";
import { runFixture } from "./pipeline";

const names = (process.argv[2] ?? "").split(",").filter(Boolean);
const withLog = process.argv.includes("--log");
const f2 = (v: number) => v.toFixed(2);
const p3 = (p: number[]) => `(${p.map((v) => v.toFixed(1)).join(",")})`;

for (const c of allCases().filter((x) => names.includes(x.name))) {
  const r = evaluateCase(c);
  const res = r.run!.result;
  const fps = c.fixture.video.fps;
  console.log(`\n=== ${c.name}  ${fps} fps  score ${r.metrics?.score.toFixed(1)}`);
  if (c.truth) {
    console.log("true touches:");
    for (const m of r.matches) {
      const tt = c.truth.touches.find((t) => t.frame === m.trueFrame)!;
      console.log(
        `  f${String(m.trueFrame).padStart(4)} ${m.trueType.padEnd(9)} ${tt.position ? p3(tt.position) : m.truePos ? p3(m.truePos) : ""}` +
        (m.found
          ? `  -> f${m.found.frame} ${m.found.label.padEnd(9)} ${p3(m.found.position)}  pos err ${m.posErrM != null ? f2(m.posErrM) : "-"} m${m.found.label === m.trueType ? "" : "  WRONG TYPE"}`
          : "  -> MISSED"),
      );
    }
    const used = new Set(r.matches.flatMap((m) => (m.found ? [m.found] : [])));
    for (const t of res.touches ?? []) if (!used.has(t)) console.log(`  FALSE  f${t.frame} ${t.label} ${p3(t.position)}`);
    if (c.truth.path) {
      console.log("path error per true flight (mean 3D m | mean |dx| |dy| |dz| | estimated share):");
      const ts = c.truth.touches;
      const last = Math.max(...Object.keys(c.truth.path).map(Number));
      for (let k = 0; k < ts.length; k++) {
        const a = ts[k].frame, b = k + 1 < ts.length ? ts[k + 1].frame : last;
        let n = 0, e = 0, dx = 0, dy = 0, dz = 0, est = 0, miss = 0;
        for (let f = a; f <= b; f++) {
          const q = c.truth.path[f], p = res.positions.get(f);
          if (!q) continue;
          if (!p) { miss++; continue; }
          n++;
          e += Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
          dx += Math.abs(p[0] - q[0]); dy += Math.abs(p[1] - q[1]); dz += Math.abs(p[2] - q[2]);
          if (res.methods.get(f) === "estimated") est++;
        }
        console.log(`  ${ts[k].type.padEnd(9)} f${a}-${b}  ${n ? f2(e / n) : "-"} | ${n ? `${f2(dx / n)} ${f2(dy / n)} ${f2(dz / n)}` : "-"} | est ${n ? f2(est / n) : "-"}${miss ? `  no path on ${miss} frames` : ""}`);
      }
    }
  }
  // The solver's own log: rally plans and segment verdicts.
  const lines: string[] = [];
  const log = console.log, warn = console.warn;
  console.log = (...a: unknown[]) => { lines.push(a.join(" ")); };
  console.warn = console.log;
  try { runFixture(c.fixture, undefined, true, c.truth?.court); } finally { console.log = log; console.warn = warn; }
  for (const l of lines) {
    if (withLog ? !l.startsWith("[ballRecon]") : /^\[rally\]|seg\[|touch splits|off-screen|line fill/.test(l)) console.log(l.replace(/  \|  /g, "\n    | "));
  }
  for (const s of res.touchSplits ?? []) {
    console.log(`  split f${s.frameBefore}-${s.frameAfter} dv ${s.dvMps.toFixed(1)} m/s cands ${s.candidates} ${s.kept ? "kept" : "dropped"}`);
  }
}
