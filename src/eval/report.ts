import * as fs from "node:fs";
import type { CaseResult } from "./evaluate";

const TOUCH_COLORS: Record<string, string> = {
  toss: "#7fb3e6", serve: "#2f86d6", reception: "#4caf50", set: "#e0b800",
  attack: "#e2463f", block: "#9b42b5", dig: "#ff8f00", touch: "#888",
};
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
const fmt = (v: number | null | undefined, d = 2) => (v == null || !Number.isFinite(v) ? "–" : v.toFixed(d));

/** Top view of one case, camera at the bottom: truth hollow, found filled. */
function courtSvg(r: CaseResult): string {
  const S = 9; // px per metre
  const X = (x: number) => 110 + x * S;
  const Y = (z: number) => 190 + z * S;
  const parts: string[] = [
    `<rect x="${X(-12)}" y="${Y(-20)}" width="${24 * S}" height="${40 * S}" fill="var(--surface)"/>`,
    `<rect x="${X(-7.5)}" y="${Y(-14)}" width="${15 * S}" height="${28 * S}" fill="var(--free)"/>`,
    `<rect x="${X(-4.5)}" y="${Y(-9)}" width="${9 * S}" height="${18 * S}" fill="var(--court)" stroke="#fff" stroke-width="1"/>`,
    `<line x1="${X(-5.5)}" y1="${Y(0)}" x2="${X(5.5)}" y2="${Y(0)}" stroke="var(--fg)" stroke-width="2.5"/>`,
    `<line x1="${X(-4.5)}" y1="${Y(-3)}" x2="${X(4.5)}" y2="${Y(-3)}" stroke="#fff" stroke-width="0.7"/>`,
    `<line x1="${X(-4.5)}" y1="${Y(3)}" x2="${X(4.5)}" y2="${Y(3)}" stroke="#fff" stroke-width="0.7"/>`,
  ];
  const truth = r.case.truth;
  if (truth?.path) {
    const pts = Object.keys(truth.path).map(Number).sort((a, b) => a - b).map((f) => truth.path![f]);
    parts.push(`<polyline points="${pts.map((p) => `${X(p[0]).toFixed(1)},${Y(p[2]).toFixed(1)}`).join(" ")}" fill="none" stroke="var(--truth)" stroke-width="1.5" stroke-dasharray="3 3"/>`);
  }
  if (r.run) {
    const frames = [...r.run.result.positions.keys()].sort((a, b) => a - b);
    let seg: string[] = [];
    const flush = () => { if (seg.length > 1) parts.push(`<polyline points="${seg.join(" ")}" fill="none" stroke="var(--ball)" stroke-width="2"/>`); seg = []; };
    frames.forEach((f, i) => {
      if (i > 0 && f - frames[i - 1] > 3) flush();
      const p = r.run!.result.positions.get(f)!;
      seg.push(`${X(p[0]).toFixed(1)},${Y(p[2]).toFixed(1)}`);
    });
    flush();
  }
  for (const m of r.matches) {
    if (!m.truePos) continue;
    parts.push(`<circle cx="${X(m.truePos[0]).toFixed(1)}" cy="${Y(m.truePos[1]).toFixed(1)}" r="5" fill="none" stroke="${TOUCH_COLORS[m.trueType] ?? "#888"}" stroke-width="2"><title>true ${m.trueType} f${m.trueFrame}</title></circle>`);
  }
  for (const t of r.run?.result.touches ?? []) {
    parts.push(`<circle cx="${X(t.position[0]).toFixed(1)}" cy="${Y(t.position[2]).toFixed(1)}" r="3.5" fill="${TOUCH_COLORS[t.label] ?? "#888"}"><title>found ${t.label} f${t.frame} h=${t.position[1].toFixed(1)} m</title></circle>`);
  }
  return `<svg viewBox="0 0 220 380" width="220" height="380" role="img" aria-label="Top view of ${esc(r.case.name)}">${parts.join("")}</svg>`;
}

function touchTable(r: CaseResult): string {
  if (!r.matches.length) {
    const found = r.run?.result.touches ?? [];
    return found.length
      ? `<p class="muted">Not labelled. Found: ${found.map((t) => `${t.label} f${t.frame}`).join(", ")}</p>`
      : `<p class="muted">No touches found.</p>`;
  }
  const rows = r.matches.map((m) => `<tr>
    <td><span class="dot" style="background:${TOUCH_COLORS[m.trueType] ?? "#888"}"></span>${m.trueType}</td>
    <td class="num">${m.trueFrame}</td>
    <td>${m.found ? `${m.found.label}${m.found.label === m.trueType ? "" : " ✗"} f${m.found.frame}` : '<span class="bad">missed</span>'}</td>
    <td class="num">${fmt(m.posErrM)}</td><td class="num">${fmt(m.heightErrM)}</td></tr>`).join("");
  return `<table><thead><tr><th>True touch</th><th>Frame</th><th>Found</th><th>Pos err m</th><th>Height err m</th></tr></thead><tbody>${rows}</tbody></table>`;
}

export function writeReport(results: CaseResult[], overall: number, file: string): string {
  const rows = results.map((r) => {
    const m = r.metrics;
    const kind = r.case.synthetic ? (r.case.knownLimit ? "known limit" : "synthetic") : r.case.truth ? "real" : "real, unlabelled";
    return `<tr><td><a href="#${esc(r.case.name)}">${esc(r.case.name)}</a></td><td>${kind}</td>
      <td class="num"><b>${m ? m.score.toFixed(1) : "–"}</b></td>
      <td class="num">${m ? `${m.touchesMatched}/${m.touchesTrue}` : "–"}</td>
      <td class="num">${m ? fmt(m.typeAccuracy) : "–"}</td>
      <td class="num">${m ? fmt(m.touchPosErrM) : "–"}</td>
      <td class="num">${m ? fmt(m.pathRmsM) : "–"}</td>
      <td class="num">${m ? fmt(m.reprojMedianPx, 1) : "–"}</td></tr>`;
  }).join("");
  const cards = results.map((r) => `<section id="${esc(r.case.name)}" class="case">
      <div>${courtSvg(r)}</div>
      <div class="detail"><h3>${esc(r.case.name)} ${r.metrics ? `<span class="score">${r.metrics.score.toFixed(1)}</span>` : ""}</h3>
      ${r.case.note ? `<p class="muted">${esc(r.case.note)}</p>` : ""}${touchTable(r)}</div></section>`).join("");
  const html = `<!doctype html><html><head><meta charset="utf-8"><title>Rally eval report</title>
<style>
:root{--bg:#f5f6f2;--surface:#fff;--fg:#1d2330;--muted:#5d6676;--line:#d9dde3;--court:#c98a4b;--free:#2b5a93;--ball:#e6b31e;--truth:#1d2330;--bad:#c0392b}
@media (prefers-color-scheme:dark){:root{--bg:#12161d;--surface:#1a202a;--fg:#e6e9ee;--muted:#9aa3b2;--line:#2c3442;--court:#a8703b;--free:#1f4370;--ball:#f2c230;--truth:#e6e9ee;--bad:#ff7b6b}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0;padding:24px 20px}
main{max-width:1100px;margin:0 auto;display:flex;flex-direction:column;gap:24px}
h1{margin:0;font-size:24px}h3{margin:0 0 6px}.muted{color:var(--muted)}.bad{color:var(--bad)}
table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line)}
th{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}.num{font-variant-numeric:tabular-nums;font-family:ui-monospace,monospace}
.case{display:flex;gap:20px;flex-wrap:wrap;background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px}
.detail{flex:1;min-width:280px}.score{font-family:ui-monospace,monospace;color:var(--muted);font-weight:400}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}a{color:inherit}
</style></head><body><main>
<h1>Rally eval report</h1>
<p class="muted">${new Date().toLocaleString()} · overall score (labelled, excluding known limits): <b>${overall.toFixed(1)}</b> · top views: camera at the bottom, dashed = true path, hollow rings = true touches, dots = found touches, yellow = solved path.</p>
<table><thead><tr><th>Case</th><th>Kind</th><th>Score</th><th>Touches</th><th>Type acc</th><th>Pos err m</th><th>Path RMS m</th><th>Reproj px</th></tr></thead><tbody>${rows}</tbody></table>
${cards}
</main></body></html>`;
  fs.writeFileSync(file, html);
  return file;
}
