/**
 * Auto-detect volleyball court corners from a video frame using Hough line transform.
 * Returns step-by-step debug images showing the entire process.
 */

type HoughLine = { theta: number; rho: number; votes: number };

export interface DetectionStep {
  label: string;
  canvas: HTMLCanvasElement;
  info: string;
}

export interface DetectionDebug {
  /** Step-by-step canvases showing the detection process */
  steps: DetectionStep[];
  /** Status messages from each step */
  log: string[];
  /** Detected corners in container-normalized [0,1] coords, or null */
  corners: { x: number; y: number }[] | null;
  /** Total detection time in milliseconds */
  totalMs: number;
}

/**
 * Detect the 4 doubles court corners from a video element.
 * Returns null if detection fails.
 */
export function detectCourtCorners(
  video: HTMLVideoElement,
  containerWidth: number,
  containerHeight: number
): { x: number; y: number }[] | null {
  return detectCourtCornersDebug(video, containerWidth, containerHeight).corners;
}

/** Create a canvas copy */
function cloneCanvas(src: HTMLCanvasElement): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = src.width;
  c.height = src.height;
  c.getContext("2d")!.drawImage(src, 0, 0);
  return c;
}

/**
 * Full detection with debug info for visualization.
 */
export function detectCourtCornersDebug(
  video: HTMLVideoElement,
  containerWidth: number,
  containerHeight: number
): DetectionDebug {
  const log: string[] = [];
  const steps: DetectionStep[] = [];
  const t0 = performance.now();
  let stepStart = t0;

  /** Push a step and record how long it took */
  const pushStep = (step: DetectionStep) => {
    const elapsed = performance.now() - stepStart;
    step.info += ` [${elapsed.toFixed(0)}ms]`;
    steps.push(step);
    stepStart = performance.now();
  };

  const fail = (msg: string) => {
    log.push(msg);
    const totalMs = performance.now() - t0;
    log.push(`Total: ${totalMs.toFixed(0)}ms`);
    return { steps, log, corners: null, totalMs };
  };

  if (video.readyState < 2 || video.videoWidth === 0) {
    return fail(`FAIL: video not ready (readyState=${video.readyState}, size=${video.videoWidth}x${video.videoHeight})`);
  }

  // ─── Step 0: Capture frame ───
  const maxW = 640;
  const scale = Math.min(1, maxW / video.videoWidth);
  const w = Math.round(video.videoWidth * scale);
  const h = Math.round(video.videoHeight * scale);

  const srcCanvas = document.createElement("canvas");
  srcCanvas.width = w;
  srcCanvas.height = h;
  const srcCtx = srcCanvas.getContext("2d", { willReadFrequently: true })!;
  srcCtx.drawImage(video, 0, 0, w, h);

  const origData = srcCtx.getImageData(0, 0, w, h).data;

  pushStep({
    label: "1. Input Frame",
    canvas: cloneCanvas(srcCanvas),
    info: `${w}×${h} (from ${video.videoWidth}×${video.videoHeight})`,
  });
  log.push(`Frame: ${w}×${h} (scale=${scale.toFixed(2)})`);

  // Margins to skip scoreboard and edges
  const marginX = Math.round(w * 0.04);
  const marginTop = Math.round(h * 0.10);
  const marginBot = Math.round(h * 0.03);

  // ─── Step 1: Find court surface via largest green blob ───
  // Detect green pixels (court surface) — slightly relaxed for distant darker grass
  const isGreen = new Uint8Array(w * h);
  for (let y = marginTop; y < h - marginBot; y++) {
    for (let x = marginX; x < w - marginX; x++) {
      const i = (y * w + x) * 4;
      const r = origData[i], g = origData[i + 1], b = origData[i + 2];
      // Green must dominate, but lower absolute threshold for darker far-court grass
      if (g > r && g > b && g > 30 && (g - Math.min(r, b)) > 10) {
        isGreen[y * w + x] = 1;
      }
    }
  }

  // Morphological close (dilate then erode by 3px) to bridge small gaps
  // from white court lines so BFS can connect both halves
  const closeR = 3;
  const dilated0 = new Uint8Array(w * h);
  for (let y = marginTop; y < h - marginBot; y++) {
    for (let x = marginX; x < w - marginX; x++) {
      if (!isGreen[y * w + x]) continue;
      for (let dy = -closeR; dy <= closeR; dy++) {
        for (let dx = -closeR; dx <= closeR; dx++) {
          const ny = y + dy, nx = x + dx;
          if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
            dilated0[ny * w + nx] = 1;
          }
        }
      }
    }
  }
  // Erode back
  const closed = new Uint8Array(w * h);
  for (let y = marginTop; y < h - marginBot; y++) {
    for (let x = marginX; x < w - marginX; x++) {
      if (!dilated0[y * w + x]) continue;
      let allSet = true;
      outer: for (let dy = -closeR; dy <= closeR; dy++) {
        for (let dx = -closeR; dx <= closeR; dx++) {
          const ny = y + dy, nx = x + dx;
          if (ny < 0 || ny >= h || nx < 0 || nx >= w || !dilated0[ny * w + nx]) {
            allSet = false;
            break outer;
          }
        }
      }
      if (allSet) closed[y * w + x] = 1;
    }
  }

  // Connected component labeling — find the largest green blob
  const labels = new Int32Array(w * h); // 0 = unlabeled
  const blobSizes: Map<number, number> = new Map();
  const blobBounds: Map<number, { minX: number; minY: number; maxX: number; maxY: number }> = new Map();
  let nextLabel = 1;

  for (let y = marginTop; y < h - marginBot; y++) {
    for (let x = marginX; x < w - marginX; x++) {
      if (!closed[y * w + x] || labels[y * w + x] !== 0) continue;
      // BFS flood fill
      const label = nextLabel++;
      const queue: number[] = [y * w + x];
      let size = 0;
      let minX = x, maxX = x, minY = y, maxY = y;
      labels[y * w + x] = label;
      while (queue.length > 0) {
        const idx = queue.pop()!;
        const px = idx % w, py = (idx - px) / w;
        size++;
        if (px < minX) minX = px;
        if (px > maxX) maxX = px;
        if (py < minY) minY = py;
        if (py > maxY) maxY = py;
        // 4-connected neighbors
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
          const nx = px + dx, ny = py + dy;
          if (nx >= marginX && nx < w - marginX && ny >= marginTop && ny < h - marginBot) {
            const ni = ny * w + nx;
            if (closed[ni] && labels[ni] === 0) {
              labels[ni] = label;
              queue.push(ni);
            }
          }
        }
      }
      blobSizes.set(label, size);
      blobBounds.set(label, { minX, minY, maxX, maxY });
    }
  }

  // Pick the largest blob
  let bestLabel = 0, bestSize = 0;
  for (const [label, size] of blobSizes) {
    if (size > bestSize) { bestSize = size; bestLabel = label; }
  }

  if (bestLabel === 0) {
    return fail("FAIL: no green region found");
  }

  const courtBounds = blobBounds.get(bestLabel)!;
  log.push(`Court blob: ${bestSize} px, bounds=(${courtBounds.minX},${courtBounds.minY})-(${courtBounds.maxX},${courtBounds.maxY})`);

  // Create court-only mask from the largest blob, then dilate to capture line edges
  const courtMask = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    if (labels[i] === bestLabel) courtMask[i] = 1;
  }

  // Dilate court mask generously to include court lines at the edges.
  // The far baseline especially needs this — the green blob may end just
  // inside the baseline, and we need the dilation to reach the line itself.
  const dilateR = 22;
  const courtDilated = new Uint8Array(w * h);
  for (let y = courtBounds.minY - dilateR; y <= courtBounds.maxY + dilateR; y++) {
    for (let x = courtBounds.minX - dilateR; x <= courtBounds.maxX + dilateR; x++) {
      if (y < 0 || y >= h || x < 0 || x >= w) continue;
      if (courtMask[y * w + x]) {
        for (let dy = -dilateR; dy <= dilateR; dy++) {
          for (let dx = -dilateR; dx <= dilateR; dx++) {
            const ny = y + dy, nx = x + dx;
            if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
              courtDilated[ny * w + nx] = 1;
            }
          }
        }
      }
    }
  }

  // Visualize court region
  const regionCanvas = document.createElement("canvas");
  regionCanvas.width = w;
  regionCanvas.height = h;
  const regionCtx = regionCanvas.getContext("2d")!;
  const regionImgData = regionCtx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const idx = i * 4;
    if (courtMask[i]) {
      regionImgData.data[idx] = 0; regionImgData.data[idx + 1] = 180; regionImgData.data[idx + 2] = 0; regionImgData.data[idx + 3] = 255;
    } else if (courtDilated[i]) {
      regionImgData.data[idx] = 0; regionImgData.data[idx + 1] = 80; regionImgData.data[idx + 2] = 0; regionImgData.data[idx + 3] = 255;
    } else if (isGreen[i]) {
      regionImgData.data[idx] = 40; regionImgData.data[idx + 1] = 40; regionImgData.data[idx + 2] = 0; regionImgData.data[idx + 3] = 255;
    } else {
      regionImgData.data[idx] = 20; regionImgData.data[idx + 1] = 20; regionImgData.data[idx + 2] = 20; regionImgData.data[idx + 3] = 255;
    }
  }
  regionCtx.putImageData(regionImgData, 0, 0);
  // Draw bounding box
  regionCtx.strokeStyle = "#ff0";
  regionCtx.lineWidth = 2;
  regionCtx.strokeRect(courtBounds.minX, courtBounds.minY, courtBounds.maxX - courtBounds.minX, courtBounds.maxY - courtBounds.minY);

  pushStep({
    label: "2. Court Region (Largest Blob)",
    canvas: regionCanvas,
    info: `Blob: ${bestSize} px | ${blobSizes.size} total blobs | Bright=court, dim=dilated, dark=other green`,
  });

  // ─── Step 2: White line isolation ───
  // Pipeline: grayscale → Gaussian blur → sharpen → adaptive contrast → line pixels
  // This isolates white court markings and rejects players, shadows, mowing patterns.

  const ey0 = Math.max(2, courtBounds.minY - dilateR - 2);
  const ey1 = Math.min(h - 3, courtBounds.maxY + dilateR + 2);
  const ex0 = Math.max(2, courtBounds.minX - dilateR - 2);
  const ex1 = Math.min(w - 3, courtBounds.maxX + dilateR + 2);

  // 2a. Grayscale
  const gray = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const r = origData[i * 4], g = origData[i * 4 + 1], b = origData[i * 4 + 2];
    gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }

  // 2b. Gaussian blur (5×5 kernel, σ≈1)
  const k5 = [1, 4, 6, 4, 1]; // 1D kernel (sum=16)
  const kSum = 256; // 16*16 for 2D
  const blurred = new Float32Array(w * h);
  const blurTmp = new Float32Array(w * h);
  // Horizontal pass
  for (let y = ey0; y <= ey1; y++) {
    for (let x = ex0; x <= ex1; x++) {
      let sum = 0;
      for (let k = -2; k <= 2; k++) {
        const nx = Math.min(w - 1, Math.max(0, x + k));
        sum += gray[y * w + nx] * k5[k + 2];
      }
      blurTmp[y * w + x] = sum;
    }
  }
  // Vertical pass
  for (let y = ey0; y <= ey1; y++) {
    for (let x = ex0; x <= ex1; x++) {
      let sum = 0;
      for (let k = -2; k <= 2; k++) {
        const ny = Math.min(h - 1, Math.max(0, y + k));
        sum += blurTmp[ny * w + x] * k5[k + 2];
      }
      blurred[y * w + x] = sum / kSum;
    }
  }

  // 2c. Unsharp mask: sharpened = gray + alpha * (gray - blurred)
  const alpha = 2.0;
  const sharpened = new Float32Array(w * h);
  for (let y = ey0; y <= ey1; y++) {
    for (let x = ex0; x <= ex1; x++) {
      const idx = y * w + x;
      sharpened[idx] = Math.min(255, Math.max(0, gray[idx] + alpha * (gray[idx] - blurred[idx])));
    }
  }

  // Visualize: grayscale + sharpened
  const sharpCanvas = document.createElement("canvas");
  sharpCanvas.width = w;
  sharpCanvas.height = h;
  const sharpCtx = sharpCanvas.getContext("2d")!;
  const sharpImg = sharpCtx.createImageData(w, h);
  for (let y = ey0; y <= ey1; y++) {
    for (let x = ex0; x <= ex1; x++) {
      if (!courtDilated[y * w + x]) continue;
      const v = Math.round(sharpened[y * w + x]);
      const idx = (y * w + x) * 4;
      sharpImg.data[idx] = v; sharpImg.data[idx + 1] = v; sharpImg.data[idx + 2] = v; sharpImg.data[idx + 3] = 255;
    }
  }
  sharpCtx.putImageData(sharpImg, 0, 0);

  pushStep({
    label: "3. Sharpened Grayscale",
    canvas: sharpCanvas,
    info: `Gaussian blur 5×5 → unsharp mask (α=${alpha})`,
  });

  // 2d. Adaptive local contrast — for each pixel, compare to local average
  // in a window. White court lines are significantly brighter than their
  // local neighbourhood. This eliminates globally-bright areas (sky, stands)
  // and isolates only local bright features (lines).
  const winR = 12; // half-window size for local mean
  // Precompute integral image of sharpened for fast local mean
  const integral = new Float64Array((w + 1) * (h + 1));
  for (let y = 1; y <= h; y++) {
    let rowSum = 0;
    for (let x = 1; x <= w; x++) {
      rowSum += sharpened[(y - 1) * w + (x - 1)];
      integral[y * (w + 1) + x] = integral[(y - 1) * (w + 1) + x] + rowSum;
    }
  }

  function localMean(cx: number, cy: number, r: number): number {
    const y0 = Math.max(0, cy - r), y1 = Math.min(h - 1, cy + r);
    const x0 = Math.max(0, cx - r), x1 = Math.min(w - 1, cx + r);
    const area = (y1 - y0 + 1) * (x1 - x0 + 1);
    if (area <= 0) return 0;
    const s = integral[(y1 + 1) * (w + 1) + (x1 + 1)]
            - integral[y0 * (w + 1) + (x1 + 1)]
            - integral[(y1 + 1) * (w + 1) + x0]
            + integral[y0 * (w + 1) + x0];
    return s / area;
  }

  // Threshold: pixel must be at least `contrastMin` brighter than local average
  // AND have absolute brightness above a minimum
  const contrastMin = 25;
  const absMin = 100;

  const linePixels: { x: number; y: number }[] = [];

  for (let y = ey0; y <= ey1; y++) {
    for (let x = ex0; x <= ex1; x++) {
      if (!courtDilated[y * w + x]) continue;
      const val = sharpened[y * w + x];
      if (val < absMin) continue;
      const lm = localMean(x, y, winR);
      if (val - lm >= contrastMin) {
        linePixels.push({ x, y });
      }
    }
  }

  // Visualize white line pixels
  const filteredCanvas = document.createElement("canvas");
  filteredCanvas.width = w;
  filteredCanvas.height = h;
  const filteredCtx = filteredCanvas.getContext("2d")!;
  filteredCtx.fillStyle = "#111";
  filteredCtx.fillRect(0, 0, w, h);
  // Mark court region faintly, then overlay line pixels
  const lineSet = new Uint8Array(w * h);
  for (const p of linePixels) lineSet[p.y * w + p.x] = 1;
  for (let y = ey0; y <= ey1; y++) {
    for (let x = ex0; x <= ex1; x++) {
      if (!courtDilated[y * w + x]) continue;
      if (lineSet[y * w + x]) {
        filteredCtx.fillStyle = "#ff3333";
        filteredCtx.fillRect(x, y, 1, 1);
      } else {
        filteredCtx.fillStyle = "rgba(0,60,0,0.3)";
        filteredCtx.fillRect(x, y, 1, 1);
      }
    }
  }

  pushStep({
    label: "4. White Line Pixels",
    canvas: filteredCanvas,
    info: `${linePixels.length} line pixels (contrast>=${contrastMin}, min brightness>=${absMin})`,
  });
  log.push(`Line pixels: ${linePixels.length}`);

  if (linePixels.length < 30) {
    return fail("FAIL: too few white line pixels (need >= 30)");
  }

  // ─── Step 3: Hough line transform ───
  const numAngles = 180;
  const diagLen = Math.ceil(Math.sqrt(w * w + h * h));
  const numRho = diagLen * 2 + 1;
  const accumulator = new Int32Array(numAngles * numRho);

  const cosT = new Float64Array(numAngles);
  const sinT = new Float64Array(numAngles);
  for (let t = 0; t < numAngles; t++) {
    const theta = (t * Math.PI) / numAngles;
    cosT[t] = Math.cos(theta);
    sinT[t] = Math.sin(theta);
  }

  // Use ALL edge pixels for Hough — subsampling loses weak lines like the far baseline
  const stride = 1;

  for (let i = 0; i < linePixels.length; i += stride) {
    const p = linePixels[i];
    for (let t = 0; t < numAngles; t++) {
      const rho = Math.round(p.x * cosT[t] + p.y * sinT[t]);
      const rIdx = rho + diagLen;
      if (rIdx >= 0 && rIdx < numRho) {
        accumulator[t * numRho + rIdx]++;
      }
    }
  }

  // Find peaks with NMS
  const peaks: HoughLine[] = [];
  const nmsR = 8;
  const votedCount = linePixels.length;
  const threshold = Math.max(votedCount * 0.008, 6);

  for (let t = 0; t < numAngles; t++) {
    for (let r = 0; r < numRho; r++) {
      const votes = accumulator[t * numRho + r];
      if (votes < threshold) continue;
      let isMax = true;
      for (let dt = -nmsR; dt <= nmsR && isMax; dt++) {
        for (let dr = -nmsR; dr <= nmsR && isMax; dr++) {
          if (dt === 0 && dr === 0) continue;
          const nt = ((t + dt) % numAngles + numAngles) % numAngles;
          const nr = r + dr;
          if (nr >= 0 && nr < numRho && accumulator[nt * numRho + nr] > votes) {
            isMax = false;
          }
        }
      }
      if (isMax) {
        peaks.push({ theta: (t * Math.PI) / numAngles, rho: r - diagLen, votes });
      }
    }
  }
  peaks.sort((a, b) => b.votes - a.votes);

  // Draw all detected lines
  const allLinesCanvas = cloneCanvas(srcCanvas);
  const allLinesCtx = allLinesCanvas.getContext("2d")!;
  for (const peak of peaks.slice(0, 40)) {
    const brightness = Math.min(255, Math.round(80 + (peak.votes / (peaks[0]?.votes || 1)) * 175));
    drawHoughLine(allLinesCtx, peak, w, h, `rgba(${brightness},${brightness},0,0.5)`, 1);
  }

  pushStep({
    label: "5. All Hough Lines",
    canvas: allLinesCanvas,
    info: `${peaks.length} line peaks (threshold=${threshold.toFixed(1)}, brighter=more votes)`,
  });
  log.push(`Hough peaks: ${peaks.length}`);

  if (peaks.length < 4) {
    return fail("FAIL: too few Hough peaks (need >= 4)");
  }

  // ─── Step 4: Classify lines by angle ───
  const halfPi = Math.PI / 2;
  const angleTol = Math.PI / 4.5; // ~40°

  const baselines: HoughLine[] = [];
  const leftLines: HoughLine[] = [];
  const rightLines: HoughLine[] = [];

  for (const peak of peaks.slice(0, 80)) {
    const fromHoriz = Math.abs(peak.theta - halfPi);
    if (fromHoriz < angleTol) {
      baselines.push(peak);
    } else if (peak.theta < halfPi) {
      leftLines.push(peak);
    } else {
      rightLines.push(peak);
    }
  }

  const classCanvas = cloneCanvas(srcCanvas);
  const classCtx = classCanvas.getContext("2d")!;
  for (const line of baselines) drawHoughLine(classCtx, line, w, h, "rgba(255,255,0,0.6)", 2);
  for (const line of leftLines) drawHoughLine(classCtx, line, w, h, "rgba(0,170,255,0.6)", 2);
  for (const line of rightLines) drawHoughLine(classCtx, line, w, h, "rgba(255,0,255,0.6)", 2);
  classCtx.font = "bold 14px sans-serif";
  classCtx.fillStyle = "#ffff00"; classCtx.fillText(`■ Baselines: ${baselines.length}`, 10, 20);
  classCtx.fillStyle = "#00aaff"; classCtx.fillText(`■ Left sides: ${leftLines.length}`, 10, 38);
  classCtx.fillStyle = "#ff00ff"; classCtx.fillText(`■ Right sides: ${rightLines.length}`, 10, 56);

  pushStep({
    label: "6. Classified by Angle",
    canvas: classCanvas,
    info: `Baselines: ${baselines.length} | Left: ${leftLines.length} | Right: ${rightLines.length}`,
  });
  log.push(`Classified: ${baselines.length} baselines, ${leftLines.length} left, ${rightLines.length} right`);

  if (baselines.length < 2 || leftLines.length < 1 || rightLines.length < 1) {
    return fail(`FAIL: need >=2 baselines, >=1 left, >=1 right`);
  }

  // ─── Step 5: Court model fitting ───
  // Instead of heuristically guessing which lines are baselines, try ALL
  // combinations of (2 baselines + 2 sidelines) and score each candidate
  // by how well the volleyball court model (attack lines) aligns with
  // detected edge pixels. The combination whose projected internal lines
  // best match the edges wins.

  const yAtCenter = (line: HoughLine) =>
    (line.rho - (w / 2) * Math.cos(line.theta)) / Math.sin(line.theta);
  const midY = h / 2;
  const xAtMid = (line: HoughLine) =>
    (line.rho - midY * Math.sin(line.theta)) / Math.cos(line.theta);

  // Volleyball court model (normalised [0,1]×[0,1])
  // y: 0=far baseline (top), 1=near baseline (bottom)
  // x: 0=left sideline, 1=right sideline
  const ATTACK_NORM = 3.0 / 18.0;  // attack line distance from net ≈ 0.1667
  const courtH = [
    { y: 0.5 - ATTACK_NORM, x0: 0, x1: 1 },  // far attack line
    { y: 0.5 + ATTACK_NORM, x0: 0, x1: 1 },  // near attack line
  ];
  const courtV: { x: number; y0: number; y1: number }[] = [];

  // Build edge proximity map — dilate edge pixels by ±proxR for fast hit testing
  const proxR = 4;
  const edgeMap = new Uint8Array(w * h);
  for (const p of linePixels) {
    for (let dy = -proxR; dy <= proxR; dy++) {
      for (let dx = -proxR; dx <= proxR; dx++) {
        const ny = p.y + dy, nx = p.x + dx;
        if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
          edgeMap[ny * w + nx] = 1;
        }
      }
    }
  }

  // Score a candidate quad by projecting internal court model lines and
  // counting how many projected sample points land near detected edge pixels.
  // Score is weighted by quad area so small accidental fits don't win.
  function scoreQuad(
    fl: { x: number; y: number }, fr: { x: number; y: number },
    nr: { x: number; y: number }, nl: { x: number; y: number }
  ): number {
    const H = computeModelToImageH(fl, fr, nr, nl);
    if (!H) return -1;

    let hits = 0, samples = 0;
    const step = 0.02;

    // Score internal horizontal lines (service lines — NOT baselines)
    for (const hline of courtH) {
      for (let mx = hline.x0; mx <= hline.x1; mx += step) {
        const p = projectPt(H, mx, hline.y);
        const px = Math.round(p.x), py = Math.round(p.y);
        samples++;
        if (px >= 0 && px < w && py >= 0 && py < h && edgeMap[py * w + px]) {
          hits++;
        }
      }
    }

    // Score internal vertical lines (none for volleyball — courtV is empty)
    for (const vline of courtV) {
      for (let my = vline.y0; my <= vline.y1; my += step) {
        const p = projectPt(H, vline.x, my);
        const px = Math.round(p.x), py = Math.round(p.y);
        samples++;
        if (px >= 0 && px < w && py >= 0 && py < h && edgeMap[py * w + px]) {
          hits++;
        }
      }
    }

    if (samples === 0) return -1;
    const hitRate = hits / samples;

    // Compute quad area (shoelface) as fraction of image area
    const pts = [fl, fr, nr, nl];
    let area2 = 0;
    for (let i = 0; i < 4; i++) {
      const j = (i + 1) % 4;
      area2 += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
    }
    const quadAreaPx = Math.abs(area2) / 2;
    const areaFrac = quadAreaPx / (w * h);

    // The real court typically covers 15-60% of the frame. Penalize tiny quads.
    // Final score = hitRate × areaWeight, where areaWeight saturates above 15%.
    const areaWeight = Math.min(1, areaFrac / 0.15);

    return hitRate * areaWeight;
  }

  // Pre-filter sidelines to remove near-duplicates
  const filteredLeft = filterCloseSidelines(leftLines, w * 0.04, midY);
  const filteredRight = filterCloseSidelines(rightLines, w * 0.04, midY);
  filteredLeft.sort((a, b) => xAtMid(a) - xAtMid(b));
  filteredRight.sort((a, b) => xAtMid(b) - xAtMid(a));

  if (filteredLeft.length < 1 || filteredRight.length < 1) {
    return fail("FAIL: not enough sidelines");
  }

  log.push(`Candidates: ${baselines.length} baselines, ${filteredLeft.length} left, ${filteredRight.length} right`);

  // Try ALL combinations of baseline pairs × sideline pairs
  type Quad = { fl: { x: number; y: number }; fr: { x: number; y: number }; nr: { x: number; y: number }; nl: { x: number; y: number } };
  let bestScore = -1;
  let bestEdges: { far: HoughLine; near: HoughLine; left: HoughLine; right: HoughLine } | null = null;
  let bestQuad: Quad | null = null;

  const maxBL = Math.min(baselines.length, 15);
  const maxSL = Math.min(filteredLeft.length, 5);
  const maxSR = Math.min(filteredRight.length, 5);
  let combosChecked = 0;

  for (let bi = 0; bi < maxBL; bi++) {
    for (let bj = bi + 1; bj < maxBL; bj++) {
      // Determine near (larger y in image) and far (smaller y)
      let nearLine = baselines[bi], farLine = baselines[bj];
      if (yAtCenter(nearLine) < yAtCenter(farLine)) {
        [nearLine, farLine] = [farLine, nearLine];
      }
      // Minimum vertical separation between baselines — real court spans >25% of frame
      if (yAtCenter(nearLine) - yAtCenter(farLine) < h * 0.25) continue;

      for (let li = 0; li < maxSL; li++) {
        for (let ri = 0; ri < maxSR; ri++) {
          const left = filteredLeft[li];
          const right = filteredRight[ri];

          const fl = intersectHoughLines(farLine, left);
          const fr = intersectHoughLines(farLine, right);
          const nr = intersectHoughLines(nearLine, right);
          const nl = intersectHoughLines(nearLine, left);
          if (!fl || !fr || !nr || !nl) continue;

          // Basic geometric validity
          if (fl.x >= fr.x || nl.x >= nr.x) continue;
          if (fl.y >= nl.y || fr.y >= nr.y) continue;
          const topW = dist(fl, fr);
          const botW = dist(nl, nr);
          if (topW > botW * 1.3) continue;

          combosChecked++;
          const score = scoreQuad(fl, fr, nr, nl);

          if (score > bestScore) {
            bestScore = score;
            bestEdges = { far: farLine, near: nearLine, left, right };
            bestQuad = { fl, fr, nr, nl };
          }
        }
      }
    }
  }

  log.push(`Court model fitting: ${combosChecked} combos tested, best=${(bestScore * 100).toFixed(1)}%`);

  if (!bestEdges || !bestQuad || bestScore < 0) {
    return fail("FAIL: no valid court quadrilateral found");
  }

  const farBaseline = bestEdges.far;
  const nearBaseline = bestEdges.near;
  const { left: leftSideline, right: rightSideline } = bestEdges;
  const { fl: farLeft, fr: farRight, nr: nearRight, nl: nearLeft } = bestQuad;

  log.push(`Best: far y=${yAtCenter(farBaseline).toFixed(0)}, near y=${yAtCenter(nearBaseline).toFixed(0)}, score=${(bestScore * 100).toFixed(1)}%`);

  const edgesCanvas = cloneCanvas(srcCanvas);
  const edgesCtx = edgesCanvas.getContext("2d")!;
  drawHoughLine(edgesCtx, farBaseline, w, h, "#ffff00", 3);
  drawHoughLine(edgesCtx, nearBaseline, w, h, "#ff8800", 3);
  drawHoughLine(edgesCtx, leftSideline, w, h, "#00aaff", 3);
  drawHoughLine(edgesCtx, rightSideline, w, h, "#ff00ff", 3);
  edgesCtx.font = "bold 13px sans-serif";
  edgesCtx.strokeStyle = "#000"; edgesCtx.lineWidth = 3;
  const labelY = (line: HoughLine) => (line.rho - (w / 2) * Math.cos(line.theta)) / Math.sin(line.theta);
  edgesCtx.strokeText("Far Baseline", w / 2 - 50, labelY(farBaseline) - 8);
  edgesCtx.fillStyle = "#ffff00"; edgesCtx.fillText("Far Baseline", w / 2 - 50, labelY(farBaseline) - 8);
  edgesCtx.strokeText("Near Baseline", w / 2 - 50, labelY(nearBaseline) + 18);
  edgesCtx.fillStyle = "#ff8800"; edgesCtx.fillText("Near Baseline", w / 2 - 50, labelY(nearBaseline) + 18);
  const lx = xAtMid(leftSideline);
  edgesCtx.strokeText("Left", lx - 10, midY); edgesCtx.fillStyle = "#00aaff"; edgesCtx.fillText("Left", lx - 10, midY);
  const rx = xAtMid(rightSideline);
  edgesCtx.strokeText("Right", rx - 15, midY); edgesCtx.fillStyle = "#ff00ff"; edgesCtx.fillText("Right", rx - 15, midY);

  pushStep({
    label: "7. Selected 4 Edges",
    canvas: edgesCanvas,
    info: `Far (yellow) | Near (orange) | Left (blue) | Right (magenta) | score=${(bestScore * 100).toFixed(1)}%`,
  });

  // ─── Step 6b: Court model overlay ───
  // Project ALL known court lines onto the frame with labels for visual verification.
  // If the algorithm is correct, every green line should sit exactly on a real white line.
  const modelCanvas = cloneCanvas(srcCanvas);
  const modelCtx = modelCanvas.getContext("2d")!;
  const bestH = computeModelToImageH(farLeft, farRight, nearRight, nearLeft);
  if (bestH) {
    // Each line has a label and color for easy verification
    const modelLines: { label: string; color: string; pts: { mx: number; my: number }[] }[] = [
      // Horizontal lines
      { label: "Far Baseline", color: "#ff4444", pts: Array.from({ length: 51 }, (_, i) => ({ mx: i / 50, my: 0 })) },
      { label: "Far Service", color: "#ffaa00", pts: Array.from({ length: 51 }, (_, i) => ({ mx: i / 50, my: 6.40 / 23.77 })) },
      { label: "Near Service", color: "#ffaa00", pts: Array.from({ length: 51 }, (_, i) => ({ mx: i / 50, my: 1 - 6.40 / 23.77 })) },
      { label: "Near Baseline", color: "#ff4444", pts: Array.from({ length: 51 }, (_, i) => ({ mx: i / 50, my: 1 })) },
      // Vertical lines
      { label: "L Doubles", color: "#44ff44", pts: Array.from({ length: 51 }, (_, i) => ({ mx: 0, my: i / 50 })) },
      { label: "L Singles", color: "#00ccff", pts: Array.from({ length: 51 }, (_, i) => ({ mx: 1.37 / 10.97, my: i / 50 })) },
      { label: "Center", color: "#ffffff", pts: Array.from({ length: 51 }, (_, i) => {
        const t = i / 50;
        return { mx: 0.5, my: 6.40 / 23.77 + t * (1 - 2 * 6.40 / 23.77) };
      }) },
      { label: "R Singles", color: "#00ccff", pts: Array.from({ length: 51 }, (_, i) => ({ mx: 1 - 1.37 / 10.97, my: i / 50 })) },
      { label: "R Doubles", color: "#44ff44", pts: Array.from({ length: 51 }, (_, i) => ({ mx: 1, my: i / 50 })) },
    ];

    for (const ml of modelLines) {
      const projected = ml.pts.map(p => projectPt(bestH, p.mx, p.my));
      modelCtx.beginPath();
      modelCtx.moveTo(projected[0].x, projected[0].y);
      for (let i = 1; i < projected.length; i++) {
        modelCtx.lineTo(projected[i].x, projected[i].y);
      }
      modelCtx.strokeStyle = ml.color;
      modelCtx.lineWidth = 2;
      modelCtx.stroke();

      // Label at midpoint
      const mid = projected[Math.floor(projected.length / 2)];
      modelCtx.font = "bold 10px sans-serif";
      modelCtx.strokeStyle = "#000";
      modelCtx.lineWidth = 3;
      modelCtx.strokeText(ml.label, mid.x + 4, mid.y - 4);
      modelCtx.fillStyle = ml.color;
      modelCtx.fillText(ml.label, mid.x + 4, mid.y - 4);
    }
  }

  pushStep({
    label: "8. Court Model Overlay",
    canvas: modelCanvas,
    info: `Green lines = projected court model (${combosChecked} combos tested)`,
  });

  // ─── Step 7: Compute corner intersections (already computed in bestQuad) ───

  const cornerCanvas = cloneCanvas(srcCanvas);
  const cornerCtx = cornerCanvas.getContext("2d")!;
  cornerCtx.globalAlpha = 0.4;
  drawHoughLine(cornerCtx, farBaseline, w, h, "#ffff00", 2);
  drawHoughLine(cornerCtx, nearBaseline, w, h, "#ff8800", 2);
  drawHoughLine(cornerCtx, leftSideline, w, h, "#00aaff", 2);
  drawHoughLine(cornerCtx, rightSideline, w, h, "#ff00ff", 2);
  cornerCtx.globalAlpha = 1.0;

  // Draw quadrilateral
  cornerCtx.beginPath();
  cornerCtx.moveTo(farLeft.x, farLeft.y);
  cornerCtx.lineTo(farRight.x, farRight.y);
  cornerCtx.lineTo(nearRight.x, nearRight.y);
  cornerCtx.lineTo(nearLeft.x, nearLeft.y);
  cornerCtx.closePath();
  cornerCtx.strokeStyle = "#00ff00";
  cornerCtx.lineWidth = 3;
  cornerCtx.stroke();

  const cColors = ["#ff4444", "#44ff44", "#4444ff", "#ffff44"];
  const cLabels = ["Far Left", "Far Right", "Near Right", "Near Left"];
  [farLeft, farRight, nearRight, nearLeft].forEach((c, i) => {
    cornerCtx.beginPath();
    cornerCtx.arc(c.x, c.y, 10, 0, Math.PI * 2);
    cornerCtx.fillStyle = cColors[i];
    cornerCtx.fill();
    cornerCtx.strokeStyle = "#fff";
    cornerCtx.lineWidth = 2;
    cornerCtx.stroke();
    cornerCtx.font = "bold 13px sans-serif";
    cornerCtx.strokeStyle = "#000"; cornerCtx.lineWidth = 3;
    cornerCtx.strokeText(cLabels[i], c.x + 14, c.y + 5);
    cornerCtx.fillStyle = "#fff";
    cornerCtx.fillText(cLabels[i], c.x + 14, c.y + 5);
  });

  pushStep({
    label: "9. Corners (Intersections)",
    canvas: cornerCanvas,
    info: `FL=(${farLeft.x.toFixed(0)},${farLeft.y.toFixed(0)}) FR=(${farRight.x.toFixed(0)},${farRight.y.toFixed(0)}) NR=(${nearRight.x.toFixed(0)},${nearRight.y.toFixed(0)}) NL=(${nearLeft.x.toFixed(0)},${nearLeft.y.toFixed(0)})`,
  });

  // ─── Validation (already done in selection loop, but double-check) ───
  const totalMs = performance.now() - t0;
  log.push(`SUCCESS — 4 court corners detected in ${totalMs.toFixed(0)}ms`);

  const videoCorners = [
    { x: farLeft.x / w, y: farLeft.y / h },
    { x: farRight.x / w, y: farRight.y / h },
    { x: nearRight.x / w, y: nearRight.y / h },
    { x: nearLeft.x / w, y: nearLeft.y / h },
  ];

  return {
    steps,
    log,
    corners: videoCorners.map((v) =>
      videoToContainer(v.x, v.y, video.videoWidth, video.videoHeight, containerWidth, containerHeight)
    ),
    totalMs,
  };
}

// ─── Helpers ───

function deg(rad: number): string {
  return (rad * 180 / Math.PI).toFixed(1) + "°";
}
// Suppress unused variable warning — used in log.push calls above
void deg;

function drawHoughLine(
  ctx: CanvasRenderingContext2D, line: HoughLine,
  w: number, h: number, color: string, lineWidth: number
) {
  const cosA = Math.cos(line.theta), sinA = Math.sin(line.theta);
  let x0: number, y0: number, x1: number, y1: number;
  if (Math.abs(sinA) > 0.001) {
    x0 = 0; y0 = line.rho / sinA;
    x1 = w; y1 = (line.rho - w * cosA) / sinA;
  } else {
    x0 = line.rho / cosA; y0 = 0;
    x1 = line.rho / cosA; y1 = h;
  }
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1);
  ctx.strokeStyle = color; ctx.lineWidth = lineWidth; ctx.stroke();
}

function intersectHoughLines(l1: HoughLine, l2: HoughLine): { x: number; y: number } | null {
  const c1 = Math.cos(l1.theta), s1 = Math.sin(l1.theta);
  const c2 = Math.cos(l2.theta), s2 = Math.sin(l2.theta);
  const det = c1 * s2 - c2 * s1;
  if (Math.abs(det) < 1e-6) return null;
  return { x: (l1.rho * s2 - l2.rho * s1) / det, y: (c1 * l2.rho - c2 * l1.rho) / det };
}

function dist(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

function filterCloseSidelines(lines: HoughLine[], minDist: number, midY: number): HoughLine[] {
  const xAt = (line: HoughLine) =>
    (line.rho - midY * Math.sin(line.theta)) / Math.cos(line.theta);
  const sorted = [...lines].sort((a, b) => xAt(a) - xAt(b));
  const result: HoughLine[] = [];
  for (const line of sorted) {
    const x = xAt(line);
    if (!result.some((k) => Math.abs(x - xAt(k)) < minDist)) {
      result.push(line);
    }
  }
  return result;
}

function videoToContainer(
  vx: number, vy: number, videoW: number, videoH: number, containerW: number, containerH: number
): { x: number; y: number } {
  const va = videoW / videoH, ca = containerW / containerH;
  let dw: number, dh: number, ox: number, oy: number;
  if (va > ca) { dw = containerW; dh = containerW / va; ox = 0; oy = (containerH - dh) / 2; }
  else { dh = containerH; dw = containerH * va; ox = (containerW - dw) / 2; oy = 0; }
  return { x: (ox + vx * dw) / containerW, y: (oy + vy * dh) / containerH };
}

/** Compute 3×3 homography mapping court model [0,1]² → image pixels.
 *  Model corners: (0,0)→fl, (1,0)→fr, (1,1)→nr, (0,1)→nl
 *  Returns a flat 9-element array [h1..h9] or null if degenerate. */
function computeModelToImageH(
  fl: { x: number; y: number }, fr: { x: number; y: number },
  nr: { x: number; y: number }, nl: { x: number; y: number }
): number[] | null {
  const a1 = fr.x - nr.x, b1 = nl.x - nr.x, c1 = nr.x - fr.x - nl.x + fl.x;
  const a2 = fr.y - nr.y, b2 = nl.y - nr.y, c2 = nr.y - fr.y - nl.y + fl.y;
  const det = a1 * b2 - a2 * b1;
  if (Math.abs(det) < 1e-10) return null;
  const h7 = (c1 * b2 - c2 * b1) / det;
  const h8 = (a1 * c2 - a2 * c1) / det;
  return [
    fr.x - fl.x + fr.x * h7,   // h1
    nl.x - fl.x + nl.x * h8,   // h2
    fl.x,                        // h3
    fr.y - fl.y + fr.y * h7,   // h4
    nl.y - fl.y + nl.y * h8,   // h5
    fl.y,                        // h6
    h7, h8, 1,                   // h7, h8, h9
  ];
}

/** Project a model-space point (mx,my) through homography H to image coordinates. */
function projectPt(H: number[], mx: number, my: number): { x: number; y: number } {
  const d = H[6] * mx + H[7] * my + H[8];
  if (Math.abs(d) < 1e-10) return { x: -9999, y: -9999 };
  return {
    x: (H[0] * mx + H[1] * my + H[2]) / d,
    y: (H[3] * mx + H[4] * my + H[5]) / d,
  };
}
