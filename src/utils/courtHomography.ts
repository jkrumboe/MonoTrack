/**
 * Minimal 4-point homography (Direct Linear Transform).
 * Computes H such that dst[i] = H * src[i] (projective, up to scale).
 */

type Vec2 = [number, number];

/** Solve an (n x n) system Ax = b via Gaussian elimination. Returns null if singular. */
function gaussElim(A: number[][], b: number[]): number[] | null {
  const n = A.length;
  const aug = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];
    if (Math.abs(aug[col][col]) < 1e-10) return null;
    for (let row = col + 1; row < n; row++) {
      const f = aug[row][col] / aug[col][col];
      for (let j = col; j <= n; j++) aug[row][j] -= f * aug[col][j];
    }
  }

  const x = new Array<number>(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    x[row] = aug[row][n];
    for (let col = row + 1; col < n; col++) x[row] -= aug[row][col] * x[col];
    x[row] /= aug[row][row];
  }
  return x;
}

/**
 * Compute 3x3 homography matrix from 4 point correspondences (DLT).
 * Returns row-major 3x3 matrix or null if degenerate.
 */
export function computeHomography(src: Vec2[], dst: Vec2[]): number[][] | null {
  if (src.length !== 4 || dst.length !== 4) return null;

  const A: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i];
    const [u, v] = dst[i];
    A.push([-x, -y, -1, 0, 0, 0, u * x, u * y]);
    b.push(-u);
    A.push([0, 0, 0, -x, -y, -1, v * x, v * y]);
    b.push(-v);
  }

  const h = gaussElim(A, b);
  if (!h) return null;

  return [
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], 1.0],
  ];
}

/** Apply a 3x3 homography to a single 2-D point. */
export function applyHomography(H: number[][], x: number, y: number): Vec2 {
  const w = H[2][0] * x + H[2][1] * y + H[2][2];
  if (Math.abs(w) < 1e-10) return [0, 0];
  return [
    (H[0][0] * x + H[0][1] * y + H[0][2]) / w,
    (H[1][0] * x + H[1][1] * y + H[1][2]) / w,
  ];
}

/**
 * Given the 4 court corner positions in image-normalised coords
 * (order: far-left, far-right, near-right, near-left — same as what
 * Python's court_model.detect() returns), compute a homography that
 * maps image positions to court-normalised coords [0,1]x[0,1] where
 * (0,0) = far-left corner and (1,1) = near-right corner.
 */
export function imageToCourtHomography(
  corners: number[][],
): number[][] | null {
  if (corners.length !== 4) return null;
  const src = corners.map((c) => [c[0], c[1]] as Vec2);
  const dst: Vec2[] = [[0, 0], [1, 0], [1, 1], [0, 1]];
  return computeHomography(src, dst);
}
