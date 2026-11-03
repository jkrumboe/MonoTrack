import { describe, expect, it } from "vitest";
import { buildMatchTimeline, nextRallyAfter, rallyAt } from "./matchTimeline";
import type { BallTouch } from "./ballReconstruction";

const FPS = 30;
const t = (s: number, label: string, z: number, y = 2): BallTouch => ({ frame: Math.round(s * FPS), position: [0, y, z], label });

/** A rally served from `side` at time `s`: toss, serve, then the other side's play. */
function rally(s: number, serverZ: number, rest: [number, string, number][] = []): BallTouch[] {
  return [t(s, "toss", serverZ), t(s + 0.8, "serve", serverZ), ...rest.map(([dt, label, z]) => t(s + dt, label, z))];
}

describe("buildMatchTimeline", () => {
  it("splits rallies at each serve and gives each point to the side that serves next", () => {
    const touches = [
      ...rally(0, 10, [[2, "reception", -6], [3, "set", -1], [4, "attack", -3]]),
      // The far side serves next: they won rally 1.
      ...rally(20, -10, [[2, "reception", 6], [3, "set", 1], [4, "attack", 3]]),
      // Near serves: they won rally 2.
      ...rally(40, 10),
    ];
    const tl = buildMatchTimeline({ touches, positions: new Map(), fps: FPS, totalFrames: 60 * FPS });
    expect(tl.rallies).toHaveLength(3);
    expect(tl.rallies.map((r) => r.server)).toEqual(["near", "far", "near"]);
    expect(tl.rallies[0].winner).toBe("far");
    expect(tl.rallies[0].winnerSource).toBe("next-serve");
    expect(tl.rallies[0].ending).toBe("kill");
    expect(tl.rallies[1].winner).toBe("near");
    expect(tl.rallies[1].ending).toBe("kill");
    expect(tl.rallies[0].touches.map((x) => x.label)).toEqual(["toss", "serve", "reception", "set", "attack"]);
  });

  it("scores the last rally by where the ball landed", () => {
    const touches = rally(0, 10, [[2, "reception", -6], [3, "set", -1], [4, "attack", -3]]);
    // Attack from the far side comes down inside the near half: far wins.
    const positions = new Map<number, [number, number, number]>([[Math.round(4.6 * FPS), [1, 0.11, 5]]]);
    const tl = buildMatchTimeline({ touches, positions, fps: FPS, totalFrames: 10 * FPS });
    expect(tl.rallies[0].winner).toBe("far");
    expect(tl.rallies[0].winnerSource).toBe("landing");
    expect(tl.rallies[0].endFrame).toBe(Math.round(4.6 * FPS));
    expect(tl.score).toEqual({ near: 0, far: 1 });
  });

  it("gives an out ball against the side that touched it last", () => {
    const touches = rally(0, 10, [[2, "reception", -6], [3, "set", -1], [4, "attack", -3]]);
    const positions = new Map<number, [number, number, number]>([[Math.round(4.6 * FPS), [6.5, 0.11, 7]]]);
    const tl = buildMatchTimeline({ touches, positions, fps: FPS, totalFrames: 10 * FPS });
    expect(tl.rallies[0].winner).toBe("near");
    expect(tl.rallies[0].ending).toBe("error");
    expect(tl.stats.far.errors).toBe(1);
  });

  it("marks aces and service errors", () => {
    const touches = [...rally(0, 10), ...rally(20, 10), ...rally(40, -10)];
    const tl = buildMatchTimeline({ touches, positions: new Map(), fps: FPS, totalFrames: 60 * FPS });
    expect(tl.rallies[0].ending).toBe("ace");
    expect(tl.rallies[1].ending).toBe("service-error");
    expect(tl.stats.near.aces).toBe(1);
    expect(tl.stats.near["service errors"]).toBe(1);
  });

  it("lets a point set by hand win over the inferred one", () => {
    const touches = [...rally(0, 10), ...rally(20, -10)];
    const first = Math.round(0 * FPS);
    const tl = buildMatchTimeline({
      touches, positions: new Map(), fps: FPS, totalFrames: 40 * FPS, pointOverrides: { [first]: "near" },
    });
    expect(tl.rallies[0].winner).toBe("near");
    expect(tl.rallies[0].winnerSource).toBe("manual");
    expect(tl.rallies[0].score).toEqual({ near: 1, far: 0 });
  });

  it("starts a new rally after a long gap even without a serve", () => {
    const touches = [...rally(0, 10, [[2, "reception", -6]]), t(30, "dig", -5), t(31, "set", -1)];
    const tl = buildMatchTimeline({ touches, positions: new Map(), fps: FPS, totalFrames: 40 * FPS });
    expect(tl.rallies).toHaveLength(2);
    expect(tl.rallies[1].server).toBeNull();
    // The second rally has no serve, so it says nothing about who won the first.
    expect(tl.rallies[0].winnerSource).toBeNull();
  });

  it("finds rallies by frame", () => {
    const touches = [...rally(10, 10), ...rally(40, -10)];
    const { rallies } = buildMatchTimeline({ touches, positions: new Map(), fps: FPS, totalFrames: 60 * FPS });
    expect(rallyAt(rallies, 10 * FPS)?.index).toBe(1);
    expect(rallyAt(rallies, 30 * FPS)).toBeNull();
    expect(nextRallyAfter(rallies, 30 * FPS)?.index).toBe(2);
    expect(nextRallyAfter(rallies, 50 * FPS)).toBeNull();
  });
});
