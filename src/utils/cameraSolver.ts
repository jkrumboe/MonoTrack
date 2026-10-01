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
  // Bounds only exclude physically absurd cameras. They were previously tight
  // enough that solutions came back sitting exactly on the cz limit, which is a
  // sign the optimiser was pinned rather than converged.
  if (cy < 1 || cy > 60) return 1e9;
  if (fov < 3 || fov > 90) return 1e9;
  if (cz < 5 || cz > 200) return 1e9;
  if (Math.abs(cx) > 40) return 1e9;

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

// ---------------------------------------------------------------------------
// Planar (homography) calibration
//
// Every court keypoint lies on the ground plane y=0. DLT for a full 3x4
// projection matrix is degenerate for coplanar points — the linear system is
// rank-deficient, so it returns an arbitrary member of a family of solutions.
// That is what produced calibrations like fov=5.2deg at z=80m with a 141px
// reprojection error. For a planar target the correct route is a 3x3
// homography plus Zhang-style decomposition, which is well conditioned.
// ---------------------------------------------------------------------------

/** Estimate the 3x3 homography mapping world (x, z) on plane y=0 to image pixels. */
function homographyFromPlanar(ptsWorldXZ: number[][], pts2d: number[][]): number[][] | null {
  const n = ptsWorldXZ.length;
  if (n < 4) return null;

  // Isotropic normalisation of both point sets (Hartley), for conditioning.
  const norm = (pts: number[][]) => {
    let mx = 0, my = 0;
    for (const [a, b] of pts) { mx += a; my += b; }
    mx /= pts.length; my /= pts.length;
    let d = 0;
    for (const [a, b] of pts) d += Math.hypot(a - mx, b - my);
    d /= pts.length;
    const sc = d > 1e-12 ? Math.SQRT2 / d : 1;
    return {
      sc, mx, my,
      out: pts.map(([a, b]) => [(a - mx) * sc, (b - my) * sc]),
    };
  };

  const W = norm(ptsWorldXZ);
  const I = norm(pts2d);

  const A: number[][] = [];
  for (let i = 0; i < n; i++) {
    const [X, Y] = W.out[i];
    const [u, v] = I.out[i];
    A.push([-X, -Y, -1, 0, 0, 0, u * X, u * Y, u]);
    A.push([0, 0, 0, -X, -Y, -1, v * X, v * Y, v]);
  }

  const h = svdNullVector(A);
  if (!h || h.length !== 9) return null;

  const Hn = [
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], h[8]],
  ];

  // Undo normalisation: H = Ti^-1 * Hn * Tw
  const Tw = [[W.sc, 0, -W.sc * W.mx], [0, W.sc, -W.sc * W.my], [0, 0, 1]];
  const TiInv = [[1 / I.sc, 0, I.mx], [0, 1 / I.sc, I.my], [0, 0, 1]];
  const mul = (a: number[][], b: number[][]) => {
    const r = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        for (let k = 0; k < 3; k++) r[i][j] += a[i][k] * b[k][j];
    return r;
  };
  return mul(TiInv, mul(Hn, Tw));
}

/**
 * Decompose a planar homography into camera pose and focal length.
 *
 * Assumes square pixels and a centred principal point, leaving focal length as
 * the only intrinsic unknown. The two orthonormality constraints on the first
 * two columns of K^-1 H each yield an estimate of f^2; whichever is better
 * conditioned is used.
 */
function calibrationFromHomography(
  H: number[][],
  imgW: number,
  imgH: number,
): CalibrationResult | null {
  // Move the principal point to the image centre.
  const cx = imgW / 2, cy = imgH / 2;
  const h = [
    [H[0][0] - cx * H[2][0], H[0][1] - cx * H[2][1], H[0][2] - cx * H[2][2]],
    [H[1][0] - cy * H[2][0], H[1][1] - cy * H[2][1], H[1][2] - cy * H[2][2]],
    [H[2][0], H[2][1], H[2][2]],
  ];

  const h11 = h[0][0], h12 = h[0][1];
  const h21 = h[1][0], h22 = h[1][1];
  const h31 = h[2][0], h32 = h[2][1];

  // r1 . r2 = 0  ->  f^2 = -(h11*h12 + h21*h22) / (h31*h32)
  const denA = h31 * h32;
  const f2a = Math.abs(denA) > 1e-14 ? -(h11 * h12 + h21 * h22) / denA : NaN;
  // |r1| = |r2|  ->  f^2 = ((h11^2+h21^2) - (h12^2+h22^2)) / (h32^2 - h31^2)
  const denB = h32 * h32 - h31 * h31;
  const f2b = Math.abs(denB) > 1e-14
    ? ((h11 * h11 + h21 * h21) - (h12 * h12 + h22 * h22)) / denB
    : NaN;

  const cands = [f2a, f2b].filter((v) => Number.isFinite(v) && v > 1);
  if (cands.length === 0) return null;
  // Prefer the better-conditioned constraint; average when both agree.
  let f2: number;
  if (cands.length === 2 && Math.abs(cands[0] - cands[1]) / Math.max(...cands) < 0.5) {
    f2 = (cands[0] + cands[1]) / 2;
  } else {
    f2 = Math.abs(denA) > Math.abs(denB) ? cands[0] : cands[cands.length - 1];
  }
  const f = Math.sqrt(f2);
  if (!Number.isFinite(f) || f <= 0) return null;

  // K^-1 applied to the columns of h.
  const kinv = (c: number[]) => [c[0] / f, c[1] / f, c[2]];
  const c1 = kinv([h[0][0], h[1][0], h[2][0]]);
  const c2 = kinv([h[0][1], h[1][1], h[2][1]]);
  const c3 = kinv([h[0][2], h[1][2], h[2][2]]);

  const n1 = Math.hypot(c1[0], c1[1], c1[2]);
  const n2 = Math.hypot(c2[0], c2[1], c2[2]);
  if (n1 < 1e-12 || n2 < 1e-12) return null;
  let lambda = 2 / (n1 + n2);   // average the two scale estimates

  // World y=0, so H's columns correspond to the world x and z axes.
  let r1 = c1.map((v) => v * lambda);
  let r3 = c2.map((v) => v * lambda);
  let t  = c3.map((v) => v * lambda);

  // The court must be in front of the camera; flip the sign if not.
  if (t[2] < 0) {
    lambda = -lambda;
    r1 = c1.map((v) => v * lambda);
    r3 = c2.map((v) => v * lambda);
    t  = c3.map((v) => v * lambda);
  }

  // Right-handed frame: columns are (world x, world y, world z).
  const r2: number[] = [
    r3[1] * r1[2] - r3[2] * r1[1],
    r3[2] * r1[0] - r3[0] * r1[2],
    r3[0] * r1[1] - r3[1] * r1[0],
  ];

  // Re-orthonormalise R = [r1, r2, r3] (columns) via Gram-Schmidt.
  const sub = (a: number[], b: number[], k: number) => a.map((v, i) => v - k * b[i]);
  const nrm = (a: number[]) => { const m = Math.hypot(a[0], a[1], a[2]); return a.map((v) => v / m); };
  const e1 = nrm(r1);
  const e2 = nrm(sub(r2, e1, r2[0] * e1[0] + r2[1] * e1[1] + r2[2] * e1[2]));
  const e3n = sub(sub(r3, e1, r3[0] * e1[0] + r3[1] * e1[1] + r3[2] * e1[2]), e2,
                  r3[0] * e2[0] + r3[1] * e2[1] + r3[2] * e2[2]);
  const e3 = nrm(e3n);
  // R has columns e1,e2,e3 -> R[row][col]
  const R = [
    [e1[0], e2[0], e3[0]],
    [e1[1], e2[1], e3[1]],
    [e1[2], e2[2], e3[2]],
  ];

  // Camera centre C = -R^T t
  const C: [number, number, number] = [
    -(R[0][0] * t[0] + R[1][0] * t[1] + R[2][0] * t[2]),
    -(R[0][1] * t[0] + R[1][1] * t[1] + R[2][1] * t[2]),
    -(R[0][2] * t[0] + R[1][2] * t[1] + R[2][2] * t[2]),
  ];
  if (!C.every(Number.isFinite)) return null;

  // Camera looks along +z_cam, which in world coords is the third row of R.
  const lookDir = [R[2][0], R[2][1], R[2][2]];
  const lookAt: [number, number, number] = [
    C[0] + lookDir[0] * 10,
    C[1] + lookDir[1] * 10,
    C[2] + lookDir[2] * 10,
  ];

  const fov = (2 * Math.atan(imgH / (2 * f)) * 180) / Math.PI;
  if (!Number.isFinite(fov) || fov <= 0) return null;

  return { fov, position: C, lookAt };
}

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

  // Step 1: analytical seed.
  // The court is planar, so a homography is the well-conditioned route; the
  // 3x4 DLT below is only a fallback for the (unused) non-coplanar case.
  let dltSeed: number[] | null = null;
  const coplanar = modelPoints.every((p) => Math.abs(p.y) < 1e-6);

  if (coplanar) {
    const H = homographyFromPlanar(modelPoints.map((p) => [p.x, p.z]), pts2d);
    const hCal = H ? calibrationFromHomography(H, imgW, imgH) : null;
    if (hCal) {
      const { fov, position, lookAt } = hCal;
      dltSeed = [position[0], position[1], position[2], lookAt[0], lookAt[1], lookAt[2], fov];
      log.push(
        `Homography seed: fov=${fov.toFixed(1)}°, pos=[${position.map((v) => (+v).toFixed(2)).join(", ")}]`,
      );
    } else {
      log.push("Homography seed failed — falling back to search");
    }
  } else {
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

  // The analytic homography solution is exact for a planar target, so it is
  // the reference. Free refinement of all 7 parameters can wander down the
  // focal-length/distance ambiguity valley — a distant camera with a narrow
  // lens looks almost identical to a nearer one with a wide lens — and end up
  // sitting on a bound (observed: fov pinned at 3 deg with the camera 136 m
  // out). Keep the refinement only when it genuinely fits better.
  if (dltSeed) {
    const seedCost = cost(dltSeed);
    const atBound =
      bestParams[6] <= 3.05 || bestParams[6] >= 89.5 ||
      bestParams[2] >= 199 || bestParams[1] >= 59.5;
    if (atBound || !(bestCost < seedCost)) {
      log.push(
        `Refinement rejected (${atBound ? "hit a parameter bound" : "no improvement"}); ` +
        `keeping analytic homography solution.`,
      );
      bestParams = dltSeed;
      bestCost = seedCost;
    }
  }

  // Report the fit in pixels — far more interpretable than the raw NDC cost.
  const avgPxErr = Math.sqrt(bestCost / Math.max(1, ndcPoints.length)) * (imgH / 2);
  log.push(`Reprojection error: ~${avgPxErr.toFixed(1)} px/point over ${ndcPoints.length} points`);
  if (avgPxErr > 25) {
    log.push(
      `WARNING: court keypoints do not fit any physical camera (${avgPxErr.toFixed(0)} px). ` +
      `The detected court is probably wrong — recalibrate the court manually.`,
    );
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
