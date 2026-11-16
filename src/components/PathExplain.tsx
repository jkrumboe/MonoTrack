/**
 * Developer view: how the solver made the path, and why.
 *
 * Lists the path as stretches (each flight between two knots, each bridged or
 * followed hole) with the solver's verdict and its reason, says what the
 * current frame is drawn by, and lists every candidate touch it considered.
 * The same stretches colour the timeline.
 */
import { useMemo, useState } from "react";
import type { SolverTrace, TouchSplitDecision } from "../utils/ballReconstruction";
import { explainSplit, spanAt, SPAN_COLORS, SPAN_NAMES, type PathSpan } from "../utils/pathExplain";

interface Props {
  trace: SolverTrace;
  spans: PathSpan[];
  touchSplits?: TouchSplitDecision[];
  cur: number;
  onSeek: (frame: number) => void;
}

export default function PathExplain({ trace, spans, touchSplits, cur, onSeek }: Readonly<Props>) {
  const [showSplits, setShowSplits] = useState(false);
  const here = useMemo(() => spanAt(spans, cur), [spans, cur]);
  const knots = useMemo(() => trace.rallies.flatMap((r) => r.knots), [trace]);
  const knotHere = knots.find((k) => Math.abs(k.frame - cur) <= 1);
  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const s of spans) c[s.status] = (c[s.status] ?? 0) + (s.to - s.from + 1);
    return c;
  }, [spans]);
  const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;

  return (
    <div className="explain">
      <div className="explain-bar" title="Share of the path's frames by how they were made">
        {(Object.keys(SPAN_COLORS) as (keyof typeof SPAN_COLORS)[]).filter((k) => counts[k]).map((k) => (
          <i key={k} style={{ width: `${(100 * counts[k]) / total}%`, background: SPAN_COLORS[k] }} title={`${SPAN_NAMES[k]}: ${counts[k]} frames`} />
        ))}
      </div>
      <div className="explain-legend">
        {(Object.keys(SPAN_COLORS) as (keyof typeof SPAN_COLORS)[]).filter((k) => counts[k]).map((k) => (
          <span key={k}><i style={{ background: SPAN_COLORS[k] }} />{SPAN_NAMES[k]} {Math.round((100 * counts[k]) / total)}%</span>
        ))}
      </div>

      <div className="explain-here">
        <div className="explain-h">Frame {cur}</div>
        {here ? (
          <>
            <div><b style={{ color: SPAN_COLORS[here.status] }}>{here.title}</b> <span className="muted">f{here.from}–{here.to}</span></div>
            <p>{here.why}</p>
          </>
        ) : <p className="muted">No path on this frame: the ball was not tracked here and nothing nearby to join it to.</p>}
        {knotHere && (
          <p>
            <b>{knotHere.reported ? `Touch: ${knotHere.label}` : `Knot, not reported as a touch (${knotHere.label})`}</b>{" "}
            at height {knotHere.position[1].toFixed(1)} m, {Math.abs(knotHere.position[2]).toFixed(1)} m from the net. {knotHere.why}.
            {knotHere.candidates.length > 0 && <> Candidates (x, z): {knotHere.candidates.map(([x, z]) => `${x.toFixed(1)}/${z.toFixed(1)}`).join(", ")}.</>}
          </p>
        )}
      </div>

      <div className="explain-h">Stretches</div>
      <div className="explain-list">
        {spans.map((s) => (
          <button
            key={`${s.status}-${s.from}-${s.to}`}
            className={`explain-row ${here === s ? "current" : ""}`}
            onClick={() => onSeek(Math.round((s.from + s.to) / 2))}
            title={s.why}
          >
            <i style={{ background: SPAN_COLORS[s.status] }} />
            <span className="explain-frames">f{s.from}–{s.to}</span>
            <span className="explain-title">{s.title}</span>
            <span className="explain-verdict">
              {s.flight
                ? (s.flight.accepted ? `${s.flight.medianErrPx?.toFixed(1) ?? "-"} px` : (s.flight.rejectReason ?? "refused").replace(/-/g, " "))
                : SPAN_NAMES[s.status]}
            </span>
          </button>
        ))}
        {trace.skipped.map(([a, b, n]) => (
          <div key={`skip-${a}`} className="explain-row muted"><i /><span className="explain-frames">f{a}–{b}</span><span className="explain-title">ignored: {n} detections, too few to solve</span></div>
        ))}
      </div>

      {touchSplits && touchSplits.length > 0 && (
        <>
          <div className="explain-h">
            Candidate touches
            <button className="inspect-link" onClick={() => setShowSplits((v) => !v)}>
              {showSplits ? "hide" : `show ${touchSplits.length} (${touchSplits.filter((s) => s.kept).length} kept)`}
            </button>
          </div>
          {showSplits && (
            <div className="explain-list">
              {touchSplits.map((s) => (
                <button key={`${s.frameBefore}-${s.frameAfter}`} className="explain-row split" onClick={() => onSeek(s.frameAfter)}>
                  <i style={{ background: s.kept ? "#69f0ae" : "#5a6472" }} />
                  <span className="explain-frames">f{s.frameBefore}–{s.frameAfter}</span>
                  <span className="explain-title">{explainSplit(s, trace.limits)}</span>
                </button>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
