import * as THREE from "three";

/**
 * Known 3D court corner positions (FIVB volleyball court in metres).
 * Court centred at origin, Y=0 on ground plane.
 * Order: far-left, far-right, near-right, near-left
 */
const COURT_CORNERS_3D = [
  { x: -4.5, y: 0, z: -9.0 },
  { x:  4.5, y: 0, z: -9.0 },
  { x:  4.5, y: 0, z:  9.0 },
  { x: -4.5, y: 0, z:  9.0 },
];

export interface CalibrationResult {
  /** Camera position in world coords */
  position: [number, number, number];
  /** Camera lookAt target */
  lookAt: [number, number, number];
  /** Vertical field of view in degrees */
  fov: number;
}

// ---------------------------------------------------------------------------
// Tiny SVD null-vector solver (power iteration on A^T A)
// ---------------------------------------------------------------------------

function svdNullVector(A: number[][]): number[] {
  const rows = A.length;
  const cols = A[0].length;

  // Build A^T * A  (cols x cols)
  const AtA: number[][] = Array.from({ length: cols }, () => new Array(cols).fill(0));
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < cols; j++) {
      let s = 0;
      for (let k = 0; k < rows; k++) s += A[k][i] * A[k][j];
      AtA[i][j] = s;
    }
  }

  // Shift to find smallest eigenvector: M = maxDiag*I - AtA
  let maxDiag = 0;
  for (let i = 0; i < cols; i++) maxDiag = Math.max(maxDiag, AtA[i][i]);
  const M: number[][] = AtA.map((row, i) =>
    row.map((v, j) => (i === j ? maxDiag - v : -v))
  );

  // Power iteration
  let v = new Array(cols).fill(1 / Math.sqrt(cols));
  for (let iter = 0; iter < 2000; iter++) {
    const Mv = new Array(cols).fill(0);
    for (let i = 0; i < cols; i++)
      for (let j = 0; j < cols; j++) Mv[i] += M[i][j] * v[j];
    let norm = 0;
    for (const x of Mv) norm += x * x;
    norm = Math.sqrt(norm);
    if (norm < 1e-15) break;
    const newV = Mv.map(x => x / norm);
    let diff = 0;
    for (let i = 0; i < cols; i++) diff += (newV[i] - v[i]) ** 2;
    v = newV;
    if (diff < 1e-20) break;
  }
  return v;
}

// ---------------------------------------------------------------------------
// DLT: recover 3x4 projection matrix from point correspondences
// ---------------------------------------------------------------------------

function dltSolveP(pts3d: number[][], pts2d: number[][]): number[][] | null {
  const n = pts3d.length;
  if (n < 4) return null;

  // Normalise 2D
  let u0 = 0, v0 = 0;
  for (const [u, v] of pts2d) { u0 += u; v0 += v; }
  u0 /= n; v0 /= n;
  let su = 0, sv = 0;
  for (const [u, v] of pts2d) { su += Math.abs(u - u0); sv += Math.abs(v - v0); }
  su = n / (su || 1); sv = n / (sv || 1);

  // Normalise 3D
  let X0 = 0, Y0 = 0, Z0 = 0;
  for (const [X, Y, Z] of pts3d) { X0 += X; Y0 += Y; Z0 += Z; }
  X0 /= n; Y0 /= n; Z0 /= n;
  let sX = 0;
  for (const [X, Y, Z] of pts3d)
    sX += Math.sqrt((X - X0) ** 2 + (Y - Y0) ** 2 + (Z - Z0) ** 2);
  sX = (n * Math.SQRT2) / (sX || 1);

  const T3 = [
    [sX, 0, 0, -sX * X0],
    [0, sX, 0, -sX * Y0],
    [0, 0, sX, -sX * Z0],
    [0, 0, 0, 1],
  ];

  const A: number[][] = [];
  for (let i = 0; i < n; i++) {
    const [X, Y, Z] = pts3d[i];
    const [u, v] = pts2d[i];
    const Xn = sX * (X - X0), Yn = sX * (Y - Y0), Zn = sX * (Z - Z0);
    const un = su * (u - u0), vn = sv * (v - v0);
    A.push([-Xn, -Yn, -Zn, -1,   0,   0,   0,  0,  un*Xn, un*Yn, un*Zn, un]);
    A.push([  0,   0,   0,  0, -Xn, -Yn, -Zn, -1,  vn*Xn, vn*Yn, vn*Zn, vn]);
  }

  const h = svdNullVector(A);
  if (!h || h.length !== 12) return null;

  const Pn: number[][] = [
    [h[0], h[1],  h[2],  h[3]],
    [h[4], h[5],  h[6],  h[7]],
    [h[8], h[9], h[10], h[11]],
  ];

  const T2inv = [[1/su, 0, u0], [0, 1/sv, v0], [0, 0, 1]];

  // PnT3
  const PnT3: number[][] = Array.from({ length: 3 }, () => new Array(4).fill(0));
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 4; j++)
      for (let k = 0; k < 4; k++) PnT3[i][j] += Pn[i][k] * T3[k][j];

  // T2inv * PnT3
  const P: number[][] = Array.from({ length: 3 }, () => new Array(4).fill(0));
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 4; j++)
      for (let k = 0; k < 3; k++) P[i][j] += T2inv[i][k] * PnT3[k][j];

  return P;
}

function gramSchmidtQR(A: number[][]): { Q: number[][], R: number[][] } | null {
  const n = 3;
  const Q: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  const R: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  const cols = [
    [A[0][0], A[1][0], A[2][0]],
    [A[0][1], A[1][1], A[2][1]],
    [A[0][2], A[1][2], A[2][2]],
  ];
  for (let j = 0; j < n; j++) {
    let v = [...cols[j]];
    for (let k = 0; k < j; k++) {
      const dot = v[0]*Q[0][k] + v[1]*Q[1][k] + v[2]*Q[2][k];
      R[k][j] = dot;
      v = v.map((x, i) => x - dot * Q[i][k]);
    }
    const norm = Math.sqrt(v[0]**2 + v[1]**2 + v[2]**2);
    if (norm < 1e-10) return null;
    R[j][j] = norm;
    for (let i = 0; i < n; i++) Q[i][j] = v[i] / norm;
  }
  return { Q, R };
}

function solveUpperTriangular(A: number[][], b: number[]): number[] | null {
  const n = A.length;
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    if (Math.abs(A[i][i]) < 1e-12) return null;
    let s = b[i];
    for (let j = i + 1; j < n; j++) s -= A[i][j] * x[j];
    x[i] = s / A[i][i];
  }
  return x;
}

function decomposeP(P: number[][], imgH: number): {
  fov: number;
  position: [number, number, number];
  lookAt: [number, number, number];
} | null {
  const M = [
    [P[0][0], P[0][1], P[0][2]],
    [P[1][0], P[1][1], P[1][2]],
    [P[2][0], P[2][1], P[2][2]],
  ];

  // RQ via QR on M^T
  const Mt = [[M[0][0],M[1][0],M[2][0]],[M[0][1],M[1][1],M[2][1]],[M[0][2],M[1][2],M[2][2]]];
  const qr = gramSchmidtQR(Mt);
  if (!qr) return null;
  const { Q: Qt, R: Kt } = qr;

  // R = Q^T, K = R^T
  const R3 = [[Qt[0][0],Qt[1][0],Qt[2][0]],[Qt[0][1],Qt[1][1],Qt[2][1]],[Qt[0][2],Qt[1][2],Qt[2][2]]];
  const K3 = [[Kt[0][0],Kt[1][0],Kt[2][0]],[Kt[0][1],Kt[1][1],Kt[2][1]],[Kt[0][2],Kt[1][2],Kt[2][2]]];

  const sign = K3[2][2] > 0 ? 1 : -1;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { K3[i][j] *= sign; R3[i][j] *= sign; }

  const D = [Math.sign(K3[0][0]) || 1, Math.sign(K3[1][1]) || 1, Math.sign(K3[2][2]) || 1];
  for (let i = 0; i < 3; i++) {
    K3[i][0] *= D[0]; K3[i][1] *= D[1]; K3[i][2] *= D[2];
    R3[0][i] *= D[0]; R3[1][i] *= D[1]; R3[2][i] *= D[2];
  }

  const p4 = [P[0][3], P[1][3], P[2][3]];
  const t = solveUpperTriangular(K3, p4);
  if (!t) return null;

  const C: [number, number, number] = [
    -(R3[0][0]*t[0] + R3[1][0]*t[1] + R3[2][0]*t[2]),
    -(R3[0][1]*t[0] + R3[1][1]*t[1] + R3[2][1]*t[2]),
    -(R3[0][2]*t[0] + R3[1][2]*t[1] + R3[2][2]*t[2]),
  ];

  const fy = Math.abs(K3[1][1]);
  const fovRad = 2 * Math.atan(imgH / (2 * fy));
  const fov = (fovRad * 180) / Math.PI;

  const lookDir: [number, number, number] = [-R3[0][2], -R3[1][2], -R3[2][2]];
  const lookAt: [number, number, number] = [
    C[0] + lookDir[0]*10, C[1] + lookDir[1]*10, C[2] + lookDir[2]*10,
  ];

  return { fov, position: C, lookAt };
}

// ---------------------------------------------------------------------------
// Nelder-Mead refinement
// ---------------------------------------------------------------------------

function projectPoint(
  worldPt: THREE.Vector3,
  camPos: THREE.Vector3,
  lookAt: THREE.Vector3,
  fov: number,
  aspect: number
): { x: number; y: number } | null {
  const cam = new THREE.PerspectiveCamera(fov, aspect, 0.1, 500);
  cam.position.copy(camPos);
  cam.lookAt(lookAt);
  cam.updateMatrixWorld();
  cam.updateProjectionMatrix();
  const v = worldPt.clone().project(cam);
  if (v.z > 1) return null;
  return { x: v.x, y: v.y };
}

function reprojectionError(
  params: number[],
  targetNDC: { x: number; y: number }[],
  aspect: number,
  worldPoints: { x: number; y: number; z: number }[]
): number {
  const [cx, cy, cz, lx, ly, lz, fov] = params;
  if (cy < 1 || cy > 40) return 1e9;
  if (fov < 5 || fov > 80) return 1e9;
  if (cz < 5 || cz > 80) return 1e9;
  if (Math.abs(cx) > 20) return 1e9;

  const camPos = new THREE.Vector3(cx, cy, cz);
  const lookAt = new THREE.Vector3(lx, ly, lz);

  let totalErr = 0;
  for (let i = 0; i < worldPoints.length; i++) {
    const w = worldPoints[i];
    const proj = projectPoint(new THREE.Vector3(w.x, w.y, w.z), camPos, lookAt, fov, aspect);
    if (!proj) return 1e9;
    const dx = proj.x - targetNDC[i].x;
    const dy = proj.y - targetNDC[i].y;
    totalErr += dx * dx + dy * dy;
  }
  return totalErr;
}

function nelderMead(
  f: (p: number[]) => number,
  start: number[],
  scales: number[],
  maxIter = 8000,
  tol = 1e-14
): number[] {
  const n = start.length;
  const simplex: { p: number[]; v: number }[] = [];
  simplex.push({ p: [...start], v: f(start) });
  for (let i = 0; i < n; i++) {
    const p = [...start]; p[i] += scales[i];
    simplex.push({ p, v: f(p) });
  }
  const alpha = 1, gamma = 2, rho = 0.5, sigma = 0.5;
  for (let iter = 0; iter < maxIter; iter++) {
    simplex.sort((a, b) => a.v - b.v);
    if (simplex[n].v - simplex[0].v < tol) break;
    const centroid = new Array(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) centroid[j] += simplex[i].p[j];
    for (let j = 0; j < n; j++) centroid[j] /= n;
    const worst = simplex[n];
    const xr = centroid.map((c, j) => c + alpha * (c - worst.p[j]));
    const fr = f(xr);
    if (fr < simplex[0].v) {
      const xe = centroid.map((c, j) => c + gamma * (xr[j] - c));
      const fe = f(xe);
      simplex[n] = fe < fr ? { p: xe, v: fe } : { p: xr, v: fr };
    } else if (fr < simplex[n - 1].v) {
      simplex[n] = { p: xr, v: fr };
    } else {
      const xc = centroid.map((c, j) => c + rho * (worst.p[j] - c));
      const fc = f(xc);
      if (fc < worst.v) {
        simplex[n] = { p: xc, v: fc };
      } else {
        for (let i = 1; i <= n; i++) {
          for (let j = 0; j < n; j++)
            simplex[i].p[j] = simplex[0].p[j] + sigma * (simplex[i].p[j] - simplex[0].p[j]);
          simplex[i].v = f(simplex[i].p);
        }
      }
    }
  }
  simplex.sort((a, b) => a.v - b.v);
  return simplex[0].p;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function solveCameraIterative(
  ndcPoints: { x: number; y: number }[],
  aspect: number,
  worldPoints?: { x: number; y: number; z: number }[]
): { result: CalibrationResult | null; log: string[] } {
  const log: string[] = [];
  const modelPoints = worldPoints ?? COURT_CORNERS_3D;
  if (ndcPoints.length < 4) return { result: null, log: ["ERROR: Need at least 4 points"] };
  if (ndcPoints.length !== modelPoints.length) {
    return { result: null, log: ["ERROR: Point count mismatch between image and world points"] };
  }

  // Convert NDC -> pixels for DLT
  const imgH = 1080;
  const imgW = Math.round(imgH * aspect);
  const pts2d = ndcPoints.map(p => [
    ((p.x + 1) / 2) * imgW,
    ((1 - p.y) / 2) * imgH,
  ]);
  const pts3d = modelPoints.map(c => [c.x, c.y, c.z]);

  // Step 1: DLT analytical seed
  let dltSeed: number[] | null = null;
  const P = dltSolveP(pts3d, pts2d);
  if (P) {
    const decomp = decomposeP(P, imgH);
    if (decomp) {
      const { fov, position, lookAt } = decomp;
      if (fov >= 5 && fov <= 80 && position[1] >= 1 && position[1] <= 40 &&
          position[2] >= 5 && position[2] <= 80 && Math.abs(position[0]) <= 20) {
        dltSeed = [position[0], position[1], position[2], lookAt[0], lookAt[1], lookAt[2], fov];
        log.push(`DLT seed: fov=${fov.toFixed(1)}°, pos=[${position.map(v => (+v).toFixed(2)).join(", ")}]`);
      }
    }
  }

  // Step 2: Nelder-Mead refinement
  const cost = (p: number[]) => reprojectionError(p, ndcPoints, aspect, modelPoints);

  const minY = Math.min(...ndcPoints.map((p) => p.y));
  const maxY = Math.max(...ndcPoints.map((p) => p.y));
  const courtSpanY = Math.max(1e-3, Math.abs(maxY - minY));
  // Estimate FOV from how much of the vertical frame the court occupies.
  // A court spanning ~1.2 NDC units vertically at 30m needs ~20°; at 50m needs ~12°.
  const estFov = Math.max(8, Math.min(65, 45 / courtSpanY));
  const camZGuess = courtSpanY > 0.9 ? 20 : courtSpanY > 0.7 ? 28 : 38;

  const fallbackStarts: number[][] = [
    // Wide-angle range
    [0, 10, camZGuess,      0, 0,    -3, estFov],
    [0, 12, camZGuess + 5,  0, 0.5,  -5, estFov - 4],
    [0,  8, camZGuess - 5,  0, 0,    -2, estFov + 6],
    [0, 10, 20,             0, 0,    -3, 35],
    [0, 14, 28,             0, 1,    -6, 28],
    // Telephoto range (broadcast camera far back)
    [0, 10, 35,             0, 0,    -3, 14],
    [0, 12, 40,             0, 1,    -4, 10],
    [0,  8, 30,             0, 0,    -2, 18],
    [0, 15, 45,             0, 2,    -6, 12],
    [0, 10, 50,             0, 0,    -3, 8],
  ];

  const starts = dltSeed ? [dltSeed, ...fallbackStarts] : fallbackStarts;
  const scales = [1, 1, 3, 1, 0.5, 2, 5];

  let bestParams = starts[0];
  let bestCost = Infinity;

  for (const s of starts) {
    if (cost(s) >= 1e8) continue;
    const params = nelderMead(cost, s, scales);
    const c = cost(params);
    if (c < bestCost) { bestCost = c; bestParams = params; }
  }

  const [cx, cy, cz, lx, ly, lz, fov] = bestParams;
  const quality = bestCost < 1e-4 ? "excellent" : bestCost < 1e-2 ? "good" : bestCost < 0.1 ? "fair" : "poor";
  log.push(`SUCCESS: Best error=${bestCost.toExponential(2)} (${quality})`);
  log.push(`Camera pos=[${cx.toFixed(2)}, ${Math.abs(cy).toFixed(2)}, ${cz.toFixed(2)}], fov=${fov.toFixed(1)}°`);
  return {
    result: {
      position: [cx, Math.abs(cy), cz],
      lookAt: [lx, ly, lz],
      fov,
    },
    log,
  };
}
