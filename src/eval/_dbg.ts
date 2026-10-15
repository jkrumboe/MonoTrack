import { allCases } from "./dataset";
import { runFixture } from "./pipeline";
const c: any = allCases().find((x) => x.name === "noisy-glitch")!;
for (const params of [undefined, { anchorWeight: 45, anchorRobustM: 4 }]) {
  const run = runFixture(c.fixture, params as any, false)!;
  const bad: string[] = [];
  for (const [f, q] of Object.entries(c.truth.path as Record<string, number[]>)) {
    const p = run.result.positions.get(+f);
    if (!p) continue;
    const e = Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
    if (e > 2) bad.push(`${f}:${e.toFixed(1)}`);
  }
  console.log(params ? "w45" : "default", bad.length, bad.slice(0, 12).join(" "));
}
