/**
 * Runs the 3D ball solver off the UI thread. A long rally takes seconds to
 * solve, which froze the whole app when it ran in a setTimeout.
 */
import { solveBallTrajectories } from "./ballReconstruction";

export type SolverRequest = { id: number; args: Parameters<typeof solveBallTrajectories> };
export type SolverResponse =
  | { id: number; result: ReturnType<typeof solveBallTrajectories> }
  | { id: number; error: string };

self.onmessage = (e: MessageEvent<SolverRequest>) => {
  const { id, args } = e.data;
  try {
    self.postMessage({ id, result: solveBallTrajectories(...args) } satisfies SolverResponse);
  } catch (err) {
    self.postMessage({ id, error: String(err) } satisfies SolverResponse);
  }
};
