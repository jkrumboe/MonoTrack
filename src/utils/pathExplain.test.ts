import { describe, expect, it } from "vitest";
import { realCases, syntheticCases } from "../eval/dataset";
import { runFixture } from "../eval/pipeline";
import { explainReject, explainSplit, spanAt, traceSpans } from "./pathExplain";

describe("how the path was made", () => {
  const c = realCases().find((x) => x.name === "usa_jap_r4") ?? syntheticCases()[0];
  const res = runFixture(c.fixture, undefined, false, c.truth?.court)!.result;
  const trace = res.trace!;
  const spans = traceSpans(trace);

  it("accounts for every frame that has a path", () => {
    for (const f of res.positions.keys()) expect(spanAt(spans, f), `frame ${f}`).not.toBeNull();
  });

  it("agrees with how the frames were found", () => {
    // A fitted stretch holds no estimated frame, and an estimate is never called fitted.
    for (const [f, m] of res.methods) {
      const s = spanAt(spans, f)!;
      if (m === "estimated") expect(s.status, `frame ${f}`).not.toBe("fitted");
    }
  });

  it("has one more knot than flights in every rally, and every touch is a reported knot", () => {
    for (const r of trace.rallies) expect(r.knots.length).toBe(r.flights.length + 1);
    const reported = trace.rallies.flatMap((r) => r.knots.filter((k) => k.reported).map((k) => k.frame));
    expect(reported).toEqual((res.touches ?? []).map((t) => t.frame));
    for (const k of trace.rallies.flatMap((r) => r.knots)) expect(k.why.length).toBeGreaterThan(10);
  });

  it("puts the solver's verdicts into words", () => {
    expect(explainReject("median-err-too-high (15.8px)", trace.limits)).toMatch(/15\.8 px .*limit 10 px/);
    expect(explainReject("speed-too-high (52.0 m/s)", trace.limits)).toMatch(/52\.0 m\/s/);
    expect(explainSplit({ frameBefore: 41, frameAfter: 45, dvMps: 2.1, candidates: 2, kept: false }, trace.limits)).toMatch(/gap.*2\.1 m\/s.*dropped/);
    expect(explainSplit({ frameBefore: 9, frameAfter: 10, dvMps: 12, candidates: 0, kept: false, kink: 300 }, trace.limits)).toMatch(/nobody there/);
  });
});
