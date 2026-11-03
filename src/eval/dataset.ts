import * as fs from "node:fs";
import * as path from "node:path";
import { SCENARIOS } from "./scenarios";
import { generate } from "./synth";
import type { Fixture, Truth } from "./types";

export interface EvalCase {
  name: string;
  fixture: Fixture;
  /** Null for a real clip that has not been labelled yet. */
  truth: Truth | null;
  synthetic: boolean;
  /** Documents a known limit; reported but not part of the regression gate. */
  knownLimit: boolean;
  note?: string;
}

/** Repo-root eval/ directory (fixtures, labels, reports). */
export const EVAL_DIR = path.resolve(__dirname, "../../eval");

export function syntheticCases(): EvalCase[] {
  return SCENARIOS.map((sc) => {
    const { fixture, truth } = generate(sc);
    return { name: sc.name, fixture, truth, synthetic: true, knownLimit: !!sc.knownLimit, note: sc.note };
  });
}

/**
 * Real clips: eval/fixtures/<clip>.json, with eval/labels/<clip>.json when
 * labelled. EVAL_FIXTURES_DIR points at another fixture folder, to score
 * analyzer changes (a new ball tracker, say) before replacing the real ones.
 */
export function realCases(): EvalCase[] {
  const fxDir = process.env.EVAL_FIXTURES_DIR ?? path.join(EVAL_DIR, "fixtures");
  if (!fs.existsSync(fxDir)) return [];
  return fs.readdirSync(fxDir)
    .filter((f) => f.endsWith(".json"))
    .sort()
    .map((f) => {
      const fixture = JSON.parse(fs.readFileSync(path.join(fxDir, f), "utf8")) as Fixture;
      const labelPath = path.join(EVAL_DIR, "labels", f);
      const truth = fs.existsSync(labelPath) ? (JSON.parse(fs.readFileSync(labelPath, "utf8")) as Truth) : null;
      return { name: fixture.clip, fixture, truth, synthetic: false, knownLimit: false };
    });
}

export function allCases(): EvalCase[] {
  return [...syntheticCases(), ...realCases()];
}
