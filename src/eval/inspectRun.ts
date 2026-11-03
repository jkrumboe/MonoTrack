/**
 * vite-node src/eval/inspectRun.ts <out.json> <clip | --all> [params.json]
 *
 * Runs the 3D solver on one case and writes everything the inspector page
 * (inspect.html) draws: per-frame detections, camera motion, the solved path
 * reprojected into the image, arc fits, touch splits, matches against the
 * labels, and a list of likely problems tagged with the pipeline stage that
 * caused them. With --all it writes just the score of every case, to judge a
 * parameter change everywhere. Started fresh per run by inspect.ts, so solver
 * edits take effect without restarting the server.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { DEFAULT_RALLY_PARAMS } from "../utils/ballReconstruction";
import { allCases, EVAL_DIR, type EvalCase } from "./dataset";
import { aggregateScore, evaluateCase, type CaseResult, type RallyParamsOverride } from "./evaluate";
import { cameraFor, courtFromLabel, projectNorm } from "./pipeline";
import { MATCH_TOLERANCE_S } from "./score";

type V2 = [number, number];
type V3 = [number, number, number];
type Stage = "detection" | "camera" | "calibration" | "fit" | "splits" | "rules";

interface Issue {
  stage: Stage;
  severity: "high" | "medium" | "low";
  frame: number;
  /** Last frame of a run of frames (gaps, outlier runs). */
  frameEnd?: number;
  text: string;
  hint?: string;
}

const r4 = (v: number) => Math.round(v * 1e4) / 1e4;
const r2 = (v: number) => Math.round(v * 100) / 100;
const p2 = (p: number[] | undefined | null): V2 | null => (p ? [r4(p[0]), r4(p[1])] : null);
const p3 = (p: number[] | undefined | null): V3 | null => (p ? [r2(p[0]), r2(p[1]), r2(p[2])] : null);
const ndcToNorm = (u: number, v: number): V2 => [r4((u + 1) / 2), r4((1 - v) / 2)];
const median = (a: number[]) => {
  if (!a.length) return null;
  const s = [...a].sort((x, y) => x - y);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/** Court lines (world x across, y up, z along; net at z = 0), sampled so a homography maps them well. */
function courtLines(): V3[][] {
  const seg = (a: V3, b: V3, n = 12): V3[] =>
    Array.from({ length: n + 1 }, (_, i) => a.map((v, k) => v + ((b[k] - v) * i) / n) as V3);
  return [
    seg([-4.5, 0, -9], [4.5, 0, -9]), seg([4.5, 0, -9], [4.5, 0, 9]), seg([4.5, 0, 9], [-4.5, 0, 9]), seg([-4.5, 0, 9], [-4.5, 0, -9]),
    seg([-4.5, 0, 0], [4.5, 0, 0]), seg([-4.5, 0, -3], [4.5, 0, -3]), seg([-4.5, 0, 3], [4.5, 0, 3]),
    seg([-5, 2.43, 0], [5, 2.43, 0]), seg([-5, 0, 0], [-5, 2.43, 0], 2), seg([5, 0, 0], [5, 2.43, 0], 2),
  ];
}

function baselineScores(): Record<string, number> {
  const f = path.join(EVAL_DIR, "baseline.json");
  return fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, "utf8")) : {};
}

/** Runs of consecutive frames where `pred` holds, as [first, last]. */
function runs(frames: number[], pred: (f: number) => boolean): [number, number][] {
  const out: [number, number][] = [];
  let start = -1;
  for (const f of frames) {
    if (pred(f)) { if (start < 0) start = f; }
    else if (start >= 0) { out.push([start, f - 1]); start = -1; }
  }
  if (start >= 0) out.push([start, frames[frames.length - 1]]);
  return out;
}

function inspect(c: EvalCase, r: CaseResult, params: RallyParamsOverride | undefined) {
  const fx = c.fixture;
  const { width: W, height: H, fps } = fx.video;
  const aspect = W / H;
  const run = r.run;
  const cam = run ? cameraFor(run.calibration, aspect) : null;
  const proj = (p: V3): V2 | null => {
    if (!cam) return null;
    const q = projectNorm(cam, p);
    return Number.isFinite(q[0]) && Number.isFinite(q[1]) ? [r4(q[0]), r4(q[1])] : null;
  };
  const effective = params
    ? { ...DEFAULT_RALLY_PARAMS, ...params, rules: { ...DEFAULT_RALLY_PARAMS.rules, ...params.rules } }
    : DEFAULT_RALLY_PARAMS;
  const issues: Issue[] = [];
  const fsec = (f: number) => (f / fps).toFixed(2) + " s";

  // ── Per-frame record ────────────────────────────────────────────────────
  const pd = new Map((run?.result.physicsDebug ?? []).map((d) => [d.frame, d]));
  const positions = run?.result.positions ?? new Map<number, V3>();
  const frames = fx.frames.map((f) => {
    const d = pd.get(f.frame);
    const solved = positions.get(f.frame) ?? null;
    return {
      f: f.frame,
      det: p2(f.ball_img_pos),
      interp: !!f.ball_interpolated,
      conf: f.ball_conf != null ? r2(f.ball_conf) : null,
      diam: f.ball_diameter_px != null ? r2(f.ball_diameter_px) : null,
      ref: p2(f.ball_ref_pos),
      H: f.ref_from_frame?.map((v) => Math.round(v * 1e6) / 1e6) ?? null,
      players: (f.players ?? []).map((b) => b.slice(0, 4).map(r4)),
      solved: p3(solved),
      proj: solved ? proj(solved) : null,
      shadow: solved ? proj([solved[0], 0, solved[2]]) : null,
      truth: p3(c.truth?.path?.[f.frame]),
      truthProj: c.truth?.path?.[f.frame] ? proj(c.truth.path[f.frame]) : null,
      pd: d ? {
        obs: ndcToNorm(d.observedU, d.observedV),
        err: d.reprojectionErrorPx != null ? r2(d.reprojectionErrorPx) : null,
        arc: d.arcId, ok: d.arcAccepted, why: d.arcRejectReason, used: d.usedInFit,
        out: d.isOutlier, speed: d.speedMps != null ? r2(d.speedMps) : null,
      } : null,
    };
  });
  const byFrame = new Map(frames.map((f) => [f.f, f]));
  const allFrames = frames.map((f) => f.f);
  const seen = allFrames.filter((f) => byFrame.get(f)!.det && !byFrame.get(f)!.interp);
  const first = seen[0] ?? 0, last = seen[seen.length - 1] ?? 0;
  const inRally = allFrames.filter((f) => f >= first && f <= last);
  const realDet = (f: number) => { const x = byFrame.get(f); return !!x && !!x.det && !x.interp; };

  // ── Stage 1: detection ──────────────────────────────────────────────────
  const truthFrames = (c.truth?.touches ?? []).map((t) => t.frame);
  const gapRuns = runs(inRally, (f) => !realDet(f)).filter(([a, b]) => b - a + 1 >= Math.max(3, Math.round(0.2 * fps)));
  for (const [a, b] of gapRuns) {
    const hidden = truthFrames.filter((t) => t >= a - 1 && t <= b + 1);
    const filled = inRally.filter((f) => f >= a && f <= b && byFrame.get(f)!.interp).length;
    issues.push({
      stage: "detection", severity: hidden.length ? "high" : b - a + 1 >= fps * 0.5 ? "medium" : "low", frame: a, frameEnd: b,
      text: `Ball not detected for ${b - a + 1} frames (${fsec(b - a + 1)})${filled ? `, ${filled} filled by straight-line interpolation` : ""}` +
        (hidden.length ? `; hides the labelled touch at frame ${hidden.join(", ")}` : ""),
      hint: "Detector (TrackNet) lost the ball. Label these frames with label_ball.py and retrain, or check for motion blur / occlusion.",
    });
  }
  const lowConf = seen.filter((f) => (byFrame.get(f)!.conf ?? 1) < 0.15);
  if (lowConf.length > seen.length * 0.3 && seen.length) {
    issues.push({ stage: "detection", severity: "low", frame: lowConf[0],
      text: `${lowConf.length} of ${seen.length} detections have confidence below 0.15`,
      hint: "Weak detections are down-weighted in the fit; many of them means the detector is unsure on this footage." });
  }

  // ── Stage 2: camera motion ──────────────────────────────────────────────
  if (fx.cameraMoving) {
    const dropped = runs(inRally, (f) => { const x = byFrame.get(f)!; return !!x.det && !x.ref; });
    for (const [a, b] of dropped) {
      issues.push({
        stage: "camera", severity: b - a + 1 >= 5 ? "medium" : "low", frame: a, frameEnd: b,
        text: `Ball detected but dropped on ${b - a + 1} frame(s): camera motion estimate not trusted`,
        hint: "camera_motion.py rejected the homography here. Check with debug_camera.py that the court stands still.",
      });
    }
  }

  // ── Stage 3: calibration ────────────────────────────────────────────────
  const calibPts: { name: string; img: V2; reproj: V2 | null; errPx: number | null }[] = [];
  if (cam) {
    let img: V2[] = [], world: number[][] = [], names: string[] = [];
    if (c.truth?.court) {
      const cl = courtFromLabel(fx, c.truth.court);
      img = cl.img; world = cl.world;
      names = (c.truth.court.points ?? []).map((p) => p.name || `point ${p.id}`);
      if (!names.length) names = ["far-left", "far-right", "near-right", "near-left"];
    } else if (fx.courtCorners) {
      img = fx.courtCorners as V2[];
      world = [[-4.5, -9], [4.5, -9], [4.5, 9], [-4.5, 9]];
      names = ["corner (-4.5,-9)", "corner (4.5,-9)", "corner (4.5,9)", "corner (-4.5,9)"];
    }
    img.forEach((p, i) => {
      const [x, z, h] = world[i];
      const q = proj([x, h ?? 0, z]);
      const errPx = q ? Math.hypot((q[0] - p[0]) * W, (q[1] - p[1]) * H) : null;
      calibPts.push({ name: names[i] ?? `point ${i}`, img: p2(p)!, reproj: q, errPx: errPx != null ? r2(errPx) : null });
    });
    const worst = calibPts.reduce((m, p) => Math.max(m, p.errPx ?? 0), 0);
    if (worst > 6) {
      const bad = calibPts.filter((p) => (p.errPx ?? 0) > 6).map((p) => `${p.name} ${p.errPx!.toFixed(0)} px`);
      issues.push({ stage: "calibration", severity: worst > 15 ? "high" : "medium", frame: c.truth?.court?.points?.[0]?.frame ?? fx.courtFrame ?? 0,
        text: `Camera fit misses court points: ${bad.join(", ")}`,
        hint: "A misplaced court point (or lens distortion) skews every 3D position. Re-place the court in label_touches.py (k)." });
    }
  } else {
    issues.push({ stage: "calibration", severity: "high", frame: 0, text: "No camera calibration: no court found and none labelled",
      hint: "Place the court by hand in label_touches.py (k)." });
  }

  // ── Stage 4: arc fit ────────────────────────────────────────────────────
  const arcMap = new Map<number, { id: number; first: number; last: number; ok: boolean | null; why: string | null; speed: number | null; errs: number[]; n: number; outliers: number }>();
  for (const f of frames) {
    if (!f.pd || f.pd.arc == null) continue;
    let a = arcMap.get(f.pd.arc);
    if (!a) { a = { id: f.pd.arc, first: f.f, last: f.f, ok: f.pd.ok, why: f.pd.why, speed: f.pd.speed, errs: [], n: 0, outliers: 0 }; arcMap.set(f.pd.arc, a); }
    a.first = Math.min(a.first, f.f); a.last = Math.max(a.last, f.f); a.n++;
    if (f.pd.err != null) a.errs.push(f.pd.err);
    if (f.pd.out) a.outliers++;
  }
  const arcs = [...arcMap.values()].sort((a, b) => a.first - b.first).map((a) => ({
    id: a.id, first: a.first, last: a.last, ok: a.ok, why: a.why, speed: a.speed, n: a.n, outliers: a.outliers,
    medErr: a.errs.length ? r2(median(a.errs)!) : null,
  }));
  for (const a of arcs) {
    if (a.ok === false) {
      issues.push({ stage: "fit", severity: a.n >= 8 ? "medium" : "low", frame: a.first, frameEnd: a.last,
        text: `Arc ${a.id} (${a.n} obs) rejected: ${a.why ?? "unknown"}${a.medErr != null ? `, median error ${a.medErr} px` : ""}`,
        hint: a.why === "too-few-observations" ? "Too few detections between gaps to fit a flight." :
          `Fit acceptance: segMaxMedianErrPx ${effective.segMaxMedianErrPx}, segMinInlierFrac ${effective.segMinInlierFrac}.` });
    }
  }
  const outRuns = runs(allFrames, (f) => !!byFrame.get(f)!.pd?.out);
  for (const [a, b] of outRuns) {
    const errs = inRally.filter((f) => f >= a && f <= b).map((f) => byFrame.get(f)!.pd?.err ?? 0);
    issues.push({ stage: "fit", severity: b - a + 1 >= 3 ? "medium" : "low", frame: a, frameEnd: b,
      text: `${b - a + 1} detection(s) treated as outliers (up to ${Math.max(...errs).toFixed(0)} px from the fitted path)`,
      hint: `Either a wrong detection (another ball, a head) or a touch the solver did not split at. Outlier above rallyOutlierPx = ${effective.rallyOutlierPx}.` });
  }
  const uncovered = runs(inRally, (f) => realDet(f) && !byFrame.get(f)!.solved);
  for (const [a, b] of uncovered) {
    if (b - a + 1 < 3) continue;
    issues.push({ stage: "fit", severity: "medium", frame: a, frameEnd: b,
      text: `Ball seen on ${b - a + 1} frames but no 3D position solved`,
      hint: "The arc covering these frames was rejected or dropped; see the arc lane." });
  }

  // ── Stages 5 + 6: touch splits and rally rules ──────────────────────────
  const splits = run?.result.touchSplits ?? [];
  const found = run?.result.touches ?? [];
  const tol = MATCH_TOLERANCE_S * fps;
  const matchedFound = new Set(r.matches.filter((m) => m.found).map((m) => m.found!));
  for (const m of r.matches) {
    if (m.found) {
      if (m.found.label !== m.trueType) {
        issues.push({ stage: "rules", severity: "medium", frame: m.trueFrame,
          text: `Touch at frame ${m.trueFrame} labelled ${m.found.label}, should be ${m.trueType}`,
          hint: "Rally rules assign types in order (serve, reception, set, attack...). A missed or extra touch earlier shifts every label after it." });
      }
      if (m.posErrM != null && m.posErrM > 1.5) {
        issues.push({ stage: "rules", severity: m.posErrM > 3 ? "high" : "medium", frame: m.trueFrame,
          text: `${m.trueType} at frame ${m.trueFrame} placed ${m.posErrM.toFixed(1)} m from the labelled player`,
          hint: "Depth along the camera ray is the weak direction. Check the ball size (diameter) and which player anchored the touch (anchorWeight)." });
      }
      continue;
    }
    // Why was it missed? Look for the split decision nearest the true touch.
    const near = splits
      .map((s) => ({ s, d: Math.abs((s.frameBefore + s.frameAfter) / 2 - m.trueFrame) }))
      .filter((x) => x.d <= tol + 3)
      .sort((a, b) => a.d - b.d)[0]?.s;
    const around = inRally.filter((f) => Math.abs(f - m.trueFrame) <= 5);
    const seenAround = around.filter(realDet).length;
    let stage: Stage, text: string, hint: string;
    if (!near && seenAround < around.length / 2) {
      stage = "detection";
      text = `Missed ${m.trueType} at frame ${m.trueFrame}: ball seen on only ${seenAround} of ${around.length} frames around it`;
      hint = "Without the ball near the contact, the turn in its path cannot be found. Improve detection here.";
    } else if (!near) {
      stage = "splits";
      text = `Missed ${m.trueType} at frame ${m.trueFrame}: no split candidate, the path through it looked smooth`;
      hint = `A split needs a deviation over touchSplitM = ${effective.touchSplitM} m from a smooth curve and touchMinSeg = ${effective.touchMinSeg} obs per side.`;
    } else if (!near.kept) {
      stage = "splits";
      text = near.candidates === 0
        ? `Missed ${m.trueType} at frame ${m.trueFrame}: split found (Δv ${near.dvMps.toFixed(1)} m/s) but no player near the ball`
        : `Missed ${m.trueType} at frame ${m.trueFrame}: split rejected, Δv ${near.dvMps.toFixed(1)} m/s < touchMinDvMps ${effective.touchMinDvMps}`;
      hint = near.candidates === 0
        ? "Players near the ball come from the person detector; check player boxes on this frame (missing / camera not trusted)."
        : "Lowering touchMinDvMps finds this one, but check it does not add false touches on other clips (Run all).";
    } else {
      stage = "rules";
      const close = found.filter((t) => !matchedFound.has(t)).sort((a, b) => Math.abs(a.frame - m.trueFrame) - Math.abs(b.frame - m.trueFrame))[0];
      const off = close ? close.frame - m.trueFrame : null;
      text = `Missed ${m.trueType} at frame ${m.trueFrame}: split kept (frames ${near.frameBefore}-${near.frameAfter}) but no touch came out within ${MATCH_TOLERANCE_S} s` +
        (close && Math.abs(off!) <= fps * 0.5 ? `; nearest unmatched touch is a ${close.label} ${Math.abs(off!)} frames ${off! > 0 ? "late" : "early"}` : "");
      hint = close && Math.abs(off!) <= fps * 0.5
        ? "The touch was found but timed off. Usually the ball was not seen at contact, so the knot sits at the edge of the detection gap."
        : "The rally rules merged or dropped it (minimum touch interval, zone limits).";
    }
    issues.push({ stage, severity: "high", frame: m.trueFrame, text, hint });
  }
  if (c.truth) {
    for (const t of found) {
      if (matchedFound.has(t)) continue;
      const sp = splits.find((s) => s.kept && Math.abs((s.frameBefore + s.frameAfter) / 2 - t.frame) <= tol + 3);
      issues.push({ stage: sp ? "splits" : "rules", severity: "medium", frame: t.frame,
        text: `Extra ${t.label} at frame ${t.frame}: no labelled touch within ${MATCH_TOLERANCE_S} s` + (sp ? ` (split Δv ${sp.dvMps.toFixed(1)} m/s)` : ""),
        hint: sp ? "A wobble or wrong detection made a velocity jump; raising touchMinDvMps or fixing the detection removes it." : "Added by the rally rules (e.g. a serve toss or block)." });
    }
  }

  const sevRank = { high: 0, medium: 1, low: 2 };
  issues.sort((a, b) => sevRank[a.severity] - sevRank[b.severity] || a.frame - b.frame);

  const videoPath = path.join(EVAL_DIR, "videos", c.name + ".mp4");
  return {
    clip: c.name, synthetic: c.synthetic, knownLimit: c.knownLimit, note: c.note ?? null,
    hasVideo: fs.existsSync(videoPath),
    video: fx.video, cameraMoving: !!fx.cameraMoving, courtFrame: fx.courtFrame ?? null,
    labelled: !!c.truth, millis: run?.millis ?? null,
    metrics: r.metrics, baseline: baselineScores()[c.name] ?? null,
    params: effective, paramsOverridden: !!params,
    calibration: run?.calibration ?? null, calibPts,
    courtLines: cam ? courtLines().map((l) => l.map(proj)) : [],
    frames, arcs,
    splits,
    found: found.map((t) => ({ frame: t.frame, label: t.label, pos: p3(t.position), proj: proj(t.position), shadow: proj([t.position[0], 0, t.position[2]]), matched: matchedFound.has(t) })),
    truth: (c.truth?.touches ?? []).map((t, i) => {
      const m = r.matches[i];
      return { frame: t.frame, type: t.type, feet: t.feet ?? null, pos: m?.truePos ? [r2(m.truePos[0]), r2(m.truePos[1])] : null,
        foundFrame: m?.found?.frame ?? null, foundLabel: m?.found?.label ?? null, posErr: m?.posErrM != null ? r2(m.posErrM) : null };
    }),
    stats: {
      frames: fx.frames.length, rally: [first, last], detected: seen.length,
      interpolated: inRally.filter((f) => byFrame.get(f)!.interp).length,
      missing: inRally.filter((f) => !byFrame.get(f)!.det).length,
      cameraDropped: fx.cameraMoving ? inRally.filter((f) => { const x = byFrame.get(f)!; return !!x.det && !x.ref; }).length : 0,
      solved: positions.size,
      inFit: frames.filter((f) => f.pd?.used).length,
      outliers: frames.filter((f) => f.pd?.out).length,
      reprojMedian: median(frames.flatMap((f) => (f.pd?.used && f.pd.err != null ? [f.pd.err] : []))),
      arcsOk: arcs.filter((a) => a.ok).length, arcsRejected: arcs.filter((a) => a.ok === false).length,
      splitsKept: splits.filter((s) => s.kept).length, splitsRejected: splits.filter((s) => !s.kept).length,
      touchesFound: found.length, touchesTrue: c.truth?.touches.length ?? null,
    },
    issues,
  };
}

const [out, target, paramsFile] = process.argv.slice(2).filter((a) => a !== "--");
const params: RallyParamsOverride | undefined = paramsFile ? JSON.parse(fs.readFileSync(paramsFile, "utf8")) : undefined;
const cases = allCases();
if (target === "--all") {
  const base = baselineScores();
  const results = cases.map((c) => evaluateCase(c, params));
  fs.writeFileSync(out, JSON.stringify({
    overall: aggregateScore(results),
    baselineOverall: (() => {
      const s = results.filter((r) => r.metrics && !r.case.knownLimit && base[r.case.name] != null).map((r) => base[r.case.name]);
      return s.length ? s.reduce((a, b) => a + b, 0) / s.length : null;
    })(),
    cases: results.map((r) => ({
      name: r.case.name, synthetic: r.case.synthetic, knownLimit: r.case.knownLimit,
      score: r.metrics?.score ?? null, baseline: base[r.case.name] ?? null,
      touches: r.metrics ? `${r.metrics.touchesMatched}/${r.metrics.touchesTrue} (found ${r.metrics.touchesFound})` : null,
    })),
  }));
} else {
  const c = cases.find((x) => x.name === target);
  if (!c) { console.error(`No case named ${target}`); process.exit(2); }
  fs.writeFileSync(out, JSON.stringify(inspect(c, evaluateCase(c, params), params)));
}
