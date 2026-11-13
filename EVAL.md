# Rally pipeline: eval log

Status: **done, with four criteria not met** (section 4). Sections 1-2 were written before any code change and are left as they were; sections 3 onwards are the result.

## 1. Baseline (2026-11-12, before any code change)

Taken from the working tree as found (branch `feat/eval-harness-camera-motion`, commit `b189916` plus the uncommitted work on it), with `npm run eval`. Raw numbers: [eval/baselines/2026-11-12-before.json](eval/baselines/2026-11-12-before.json). `npm test` passed (54 tests).

How each number is defined:
- **Type correct** = touches found at the right time with the right type ÷ labelled touches (a missed touch counts as wrong).
- **Precision** = matched touches ÷ found touches, pooled over clips. **Recall** = matched ÷ labelled.
- **Touch position** = median over clips of each clip's mean horizontal error. Real clips: against the labelled feet on the floor.
- **Path RMS** = median over scenarios of each scenario's 3D RMS error.
- **Contact time** = mean over clips of each clip's mean |found − labelled| frame.
- **Reprojection** = median over clips of each clip's median error on fitted frames.

### Summary

| Metric | Synthetic (16 gated) | Real, all 11 | Real, tuning 8 | Real, holdout 3 |
|---|---|---|---|---|
| Mean score | 73.3 | 69.8 | 69.9 | 69.7 |
| Path RMS median / worst (m) | 1.67 / 3.61 | - | - | - |
| Touch position (m) | 0.34 | 1.89 | 1.64 | 2.18 |
| Type correct | 81% (78/96) | 76% (51/67) | 71% (29/41) | 85% (22/26) |
| Recall | 84% (81/96) | 85% (57/67) | 80% (33/41) | 92% (24/26) |
| Precision | 98% (81/83) | 73% (57/78) | 70% (33/47) | 77% (24/31) |
| Contact time mean / worst clip (frames) | 0.66 / 1.67 | 0.97 / 2.00 | 0.96 / 2.00 | 1.00 / 1.00 |
| Reprojection median / worst clip (px) | 4.5 / 6.7 | 3.3 / 6.7 | 2.6 / 6.7 | 3.6 / 5.2 |

Median 3D error on synthetic data, the reprojection coverage and per-frame trajectory coverage were not measured by the eval at this point; they are added in section 3 and their baseline values are measured there with the solver still unchanged.

### Per case

| Case | Kind | Score | Touches matched/true (found) | Type acc. of matched | Pos err m | Height err m | dt frames | Path RMS m | Reproj px |
|---|---|---|---|---|---|---|---|---|---|
| near-serve | synth | 76.7 | 5/6 (5) | 1.00 | 0.27 | 0.20 | 0.80 | 1.57 | 3.8 |
| far-serve | synth | 71.7 | 5/6 (5) | 1.00 | 0.45 | 0.24 | 1.00 | 2.82 | 4.2 |
| attacker-hidden | synth | 75.5 | 5/6 (5) | 1.00 | 0.33 | 0.10 | 0.80 | 1.62 | 4.1 |
| receiver-hidden | synth | 46.6 | 4/6 (4) | 0.50 | 2.61 | 0.36 | 0.25 | 3.61 | 2.8 |
| no-size | synth | 76.1 | 5/6 (5) | 1.00 | 0.23 | 0.11 | 0.40 | 1.78 | 4.6 |
| noisy-glitch | synth | 71.6 | 5/6 (6) | 0.80 | 0.49 | 0.15 | 0.80 | 0.73 | 6.7 |
| biased-size | synth | 75.6 | 5/6 (5) | 1.00 | 0.24 | 0.10 | 0.40 | 1.76 | 5.2 |
| high-camera | synth | 74.2 | 4/6 (4) | 1.00 | 0.29 | 0.20 | 0.25 | 0.95 | 5.0 |
| low-camera | synth | 74.5 | 5/6 (5) | 1.00 | 0.26 | 0.07 | 0.40 | 2.27 | 4.9 |
| wide-camera | synth | 78.7 | 5/6 (5) | 1.00 | 0.26 | 0.12 | 0.40 | 1.00 | 4.6 |
| offset-camera | synth | 74.6 | 5/6 (5) | 1.00 | 0.28 | 0.27 | 0.80 | 2.22 | 4.5 |
| 60fps | synth | 82.1 | 6/6 (6) | 1.00 | 0.48 | 0.06 | 0.00 | 0.81 | 3.2 |
| side-static | synth | 79.3 | 6/6 (6) | 1.00 | 0.70 | 0.25 | 1.00 | 0.72 | 4.5 |
| side-static-far | synth | 68.7 | 6/6 (6) | 1.00 | 1.25 | 0.34 | 1.67 | 1.73 | 6.5 |
| side-pan | synth | 74.5 | 5/6 (6) | 1.00 | 0.72 | 0.34 | 1.20 | 0.60 | 4.2 |
| behind-pan | synth | 73.0 | 5/6 (5) | 1.00 | 0.36 | 0.08 | 0.40 | 2.57 | 4.6 |
| side-pan-raw | limit | 33.9 | 4/6 (4) | 0.00 | 5.90 | 1.33 | 0.50 | 6.80 | 4.0 |
| back-row-attack | limit | 44.2 | 3/6 (3) | 0.67 | 2.70 | 0.49 | 0.00 | 5.44 | 2.8 |
| long-rally | limit | 63.2 | 7/8 (7) | 0.71 | 0.97 | 0.24 | 0.86 | 2.71 | 4.4 |
| pol_ita_r1 | real | 60.7 | 3/4 (6) | 1.00 | 2.40 | - | 1.00 | - | 2.5 |
| pol_ita_r2 | real | 71.6 | 5/5 (7) | 1.00 | 1.89 | - | 0.60 | - | 4.1 |
| pol_ita_r3 | real, holdout | 76.7 | 4/4 (6) | 1.00 | 1.16 | - | 1.00 | - | 3.6 |
| pol_ita_r4 | real | 78.6 | 4/4 (4) | 0.75 | 1.04 | - | 1.00 | - | 2.2 |
| pol_ita_r5 | real | 76.4 | 2/2 (3) | 1.00 | 1.06 | - | 0.50 | - | 5.2 |
| pol_ita_r6 | real, holdout | 66.2 | 8/8 (10) | 0.75 | 2.47 | - | 1.00 | - | 3.3 |
| pol_ita_r7 | real | 59.9 | 4/5 (6) | 1.00 | 2.70 | - | 0.75 | - | 6.7 |
| usa_jap_r1 | real | 75.0 | 2/3 (2) | 1.00 | 0.92 | - | 1.00 | - | 1.6 |
| usa_jap_r2 | real, holdout | 66.1 | 12/14 (15) | 1.00 | 2.18 | - | 1.00 | - | 5.2 |
| usa_jap_r3 | real | 55.7 | 11/16 (17) | 0.73 | 2.36 | - | 0.82 | - | 2.7 |
| usa_jap_r4 | real | 81.0 | 2/2 (2) | 1.00 | 1.39 | - | 2.00 | - | 1.1 |

## 2. Holdout

Chosen on 2026-11-12 before any tuning, one short, one medium and one long rally, from both matches:

- `pol_ita_r3` (4 touches)
- `pol_ita_r6` (8 touches)
- `usa_jap_r2` (14 touches)

These three are never used to choose parameters or rules. They are reported separately and must not get worse than the baseline above. The other 8 labelled clips are the tuning set. `France_vs_Japan` is unlabelled and only used by its reviewed check in `npm test`.

## 3. What was added to the eval (solver still unchanged)

- **Five tripod scenarios**, gated: a phone on a tripod behind the end line, static, 1080p, the whole court in view. `tripod-30`, `tripod-60` (7 m behind, 3.5 m up, 46° lens), `tripod-close-30` (5 m, 3 m up, 56°), `tripod-far-60` (8 m, 4 m up, 1.5 m off-centre, 42°), `tripod-long-30` (dig, set and second attack, ball lands in). The ball is ~10 px at the far baseline and ~17 px at the net; its size is given 20 % noise.
- **Metrics**: median 3D error (synthetic), type correct out of all true touches, pooled precision, line coverage (frames from the first to the last contact that have a path), reprojection coverage (detections the solver fitted, not estimated).
- **Tuning / holdout split** in every report, and `npm run eval:diag -- <clip>` / `npm run eval:splits` to see why a clip goes wrong.

Baseline with those in place, solver unchanged ([eval/baselines/2026-11-12-before-extended.json](eval/baselines/2026-11-12-before-extended.json)):

| Metric | Synthetic (21 gated) | of which tripod (5) | Real, tuning 8 | Real, holdout 3 |
|---|---|---|---|---|
| Mean score | 70.6 | 62.0 | 69.9 | 69.7 |
| Type correct | 73% | 50% (16/32) | 71% | 85% |
| Path RMS median / worst (m) | 1.71 / 3.85 | 1.71 / 3.85 | - | - |
| Median 3D error (m) | 0.43 | - | - | - |
| Line coverage, worst clip | 100% | 100% | 93% | 100% |
| Reprojection coverage | 77% | - | 79.7% | 80.7% |

The tripod scenarios were the weakest thing measured: half their touches were lost.

## 4. Result (2026-11-13)

Raw numbers: [eval/baselines/2026-11-13-after.json](eval/baselines/2026-11-13-after.json). `npm test` passes, including the gate and an acceptance test for every criterion marked met below.

### Acceptance criteria

| Criterion | Target | Baseline | Final | |
|---|---|---|---|---|
| **Synthetic, 21 gated scenarios (16 + 5 tripod)** | | | | |
| Path RMS, median | ≤ 1.0 m | 1.71 (1.67 on the 16) | **0.65** | met |
| Path RMS, worst scenario | ≤ 2.0 m | 3.85 | **2.58** (`behind-pan`) | **not met** |
| Touch error, horizontal median | ≤ 0.35 m | 0.36 (0.34 on the 16) | **0.38** (0.41 on the 16) | **not met** |
| Type correct | ≥ 90% | 73% (81% on the 16) | **97%** (124/128) | met |
| Frames between first and last contact with a path | 100% | 100% | **100%** | met |
| **Real, tuning set (8 clips)** | | | | |
| Contact time, mean / worst clip | ≤ 1.0 / ≤ 2 frames | 0.96 / 2.00 | **0.73 / 1.50** | met |
| Reprojection, median / worst clip | ≤ 3.0 / ≤ 6 px | 2.6 / 6.7 | **2.6 / 5.1** | within limits, but see coverage |
| Reprojection coverage | ≥ baseline | 79.7% | **76.3%** | **not met** |
| Type correct | ≥ 85% | 71% (29/41) | **85%** (35/41) | met |
| Touch precision | ≥ 85% | 70% (33/47) | **86%** (36/42) | met |
| Touch position, median | ≤ 1.2 m | 1.64 | **1.18** | met |
| **Real, holdout (3 clips): no metric worse** | | | | **not met** (2 of 8 worse) |
| Score | | 69.7 | 75.9 | better |
| Type correct / recall / precision | | 85% / 92% / 77% | 88% / 96% / 89% | better |
| Touch position, median | | 2.18 m | 1.61 m | better |
| Contact time, mean / worst | | 1.00 / 1.00 | 0.61 / 0.75 | better |
| Reprojection, median / worst | | 3.6 / 5.2 px | 4.1 / 4.3 px | median **worse** |
| Reprojection coverage | | 80.7% | 78.3% | **worse** |
| **Performance** | | | | |
| 10 s rally, 1080p 25 fps, warm | ≤ 48 s (stretch 30 s) | not measured before | **32.3 s** analysis + ~1.5 s solve | met; stretch missed by ~4 s |
| 20 s rally completes | yes | | **54.9 s** + 2.9 s solve (19.2 s clip) | met |
| Peak VRAM added | ≤ 4 GB | | **0.9-1.0 GB** | met |
| 3D solver, all 11 real clips | not slower | 18.4 s | **14.3 s** | met |
| **Editor** | | | | |
| Re-fit per edit | < 200 ms | 1-5 s (full solve) | **0.4-0.6 ms median, 1.5-12 ms worst** (three runs) | met |
| **General** | | | | |
| `npm test` and the eval gate | pass | pass | **pass** | met |

### The four criteria not met, and why

1. **Worst synthetic path RMS 2.58 m (`behind-pan`).** All of it is the last flight, after the block: 5.9 m off along the line of sight, while the rest of that rally is within 0.1-1.0 m. From 47 m behind the court a ball deflected off the block towards or away from the camera looks the same; only its apparent size says which, and this scenario is unchanged from its baseline (2.57 m). Every other gated scenario is now ≤ 1.65 m. It needs either a second view or a prior on where blocked balls go, which I did not want to invent without data.
2. **Synthetic touch position 0.38 m (was 0.36 on the same 21, 0.34 on the 16).** A side effect of finding more: 126 of 128 touches are found instead of 101, and the 25 new ones are the hard ones (a block 0.12 s after its attack, a hidden player), which sit 0.5-1 m off and pull each clip's mean up. Stated as a miss because the criterion is on the number as defined.
3. **Reprojection coverage 76.3% against 79.7% (tuning), 78.3% against 80.7% (holdout).** The solver now cuts flights at more (real) touches, and three clips have a fast flight between two anchored touches that it then refuses to call a fit (`pol_ita_r7` 42% → 30%, `usa_jap_r3` 91% → 82%, `pol_ita_r1` 72% → 64%). Those frames still have a path (dashed, "estimated"). I tried letting such a touch change player or drop its anchor when a flight at it is refused: coverage rose 2 points but touch positions got worse on several scenarios, so it was not kept. By the rule you set, the tuning-set reprojection error therefore does not count as met, although the numbers themselves are within the limits. A per-clip coverage floor is now in the gate so it cannot fall further.
4. **Holdout reprojection median 4.1 px against 3.6 px.** One clip (`pol_ita_r3`: 3.6 → 4.3 px, coverage 95% → 82%); the other two improved. Same cause as 3. Every other holdout metric is better than its baseline.

### Before → after, per case

| Case | Kind | Score | Touches matched/true (found) | Type correct (count) | Pos err m | dt f | Path RMS m | Reproj px | Coverage |
|---|---|---|---|---|---|---|---|---|---|
| near-serve | synth | 76.7 → 83.8 | 5/6 (5) → 6/6 (6) | 5 → 6 | 0.27 → 0.43 | 0.80 → 0.50 | 1.57 → 0.60 | 3.8 → 3.8 | 0.94 → 0.78 |
| far-serve | synth | 71.7 → 85.9 | 5/6 (5) → 6/6 (6) | 5 → 6 | 0.45 → 0.34 | 1.00 → 0.50 | 2.82 → 0.51 | 4.2 → 3.6 | 0.72 → 0.94 |
| attacker-hidden | synth | 75.5 → 83.6 | 5/6 (5) → 6/6 (6) | 5 → 6 | 0.33 → 0.42 | 0.80 → 0.50 | 1.62 → 0.66 | 4.1 → 3.6 | 0.66 → 0.78 |
| receiver-hidden | synth | 46.6 → 83.5 | 4/6 (4) → 6/6 (6) | 2 → 6 | 2.61 → 0.41 | 0.25 → 0.50 | 3.61 → 0.65 | 2.8 → 4.0 | 0.35 → 0.78 |
| no-size | synth | 76.1 → 87.0 | 5/6 (5) → 6/6 (6) | 5 → 6 | 0.23 → 0.28 | 0.40 → 0.83 | 1.78 → 0.45 | 4.6 → 4.0 | 0.97 → 0.95 |
| noisy-glitch | synth | 71.6 → 81.9 | 5/6 (6) → 6/6 (6) | 4 → 6 | 0.49 → 0.38 | 0.80 → 0.83 | 0.73 → 0.69 | 6.7 → 6.9 | 0.75 → 0.75 |
| biased-size | synth | 75.6 → 73.6 | 5/6 (5) → 6/6 (6) | 5 → 5 | 0.24 → 0.70 | 0.40 → 0.83 | 1.76 → 1.36 | 5.2 → 4.3 | 0.98 → 0.97 |
| high-camera | synth | 74.2 → 86.1 | 4/6 (4) → 6/6 (6) | 4 → 6 | 0.29 → 0.31 | 0.25 → 0.50 | 0.95 → 0.47 | 5.0 → 4.5 | 0.63 → 0.96 |
| low-camera | synth | 74.5 → 80.3 | 5/6 (5) → 6/6 (6) | 5 → 6 | 0.26 → 0.47 | 0.40 → 0.83 | 2.27 → 0.96 | 4.9 → 4.6 | 0.98 → 0.97 |
| wide-camera | synth | 78.7 → 86.5 | 5/6 (5) → 6/6 (6) | 5 → 6 | 0.26 → 0.30 | 0.40 → 0.50 | 1.00 → 0.46 | 4.6 → 4.3 | 0.98 → 0.96 |
| offset-camera | synth | 74.6 → 88.2 | 5/6 (5) → 6/6 (6) | 5 → 6 | 0.28 → 0.21 | 0.80 → 0.50 | 2.22 → 0.40 | 4.5 → 4.5 | 0.96 → 0.93 |
| 60fps | synth | 82.1 → 83.2 | 6/6 (6) → 6/6 (6) | 6 → 6 | 0.48 → 0.40 | 0.00 → 0.33 | 0.81 → 0.79 | 3.2 → 3.2 | 1.00 → 1.00 |
| side-static | synth | 79.3 → 79.9 | 6/6 (6) → 6/6 (6) | 6 → 6 | 0.70 → 0.66 | 1.00 → 0.67 | 0.72 → 0.72 | 4.5 → 4.4 | 0.94 → 0.96 |
| side-static-far | synth | 68.7 → 78.5 | 6/6 (6) → 6/6 (6) | 6 → 6 | 1.25 → 0.70 | 1.67 → 0.83 | 1.73 → 0.77 | 6.5 → 5.3 | 0.67 → 0.98 |
| side-pan | synth | 74.5 → 70.2 | 5/6 (6) → 6/6 (6) | 5 → 6 | 0.72 → 1.25 | 1.20 → 0.50 | 0.60 → 1.65 | 4.2 → 4.4 | 0.70 → 0.76 |
| behind-pan | synth | 73.0 → 72.8 | 5/6 (5) → 5/6 (5) | 5 → 5 | 0.36 → 0.37 | 0.40 → 0.40 | 2.57 → 2.58 | 4.6 → 4.6 | 0.93 → 0.93 |
| tripod-30 | synth | 59.1 → 89.0 | 2/6 (4) → 6/6 (6) | 2 → 6 | 0.31 → 0.23 | 0.00 → 0.83 | 1.71 → 0.39 | 2.6 → 3.1 | 0.42 → 0.98 |
| tripod-60 | synth | 53.0 → 79.4 | 3/6 (6) → 6/6 (6) | 2 → 5 | 0.72 → 0.64 | 1.33 → 0.33 | 1.67 → 0.65 | 3.3 → 2.5 | 0.63 → 1.00 |
| tripod-close-30 | synth | 70.9 → 77.8 | 5/6 (5) → 5/6 (5) | 4 → 5 | 0.39 → 0.28 | 1.60 → 0.40 | 2.17 → 1.20 | 3.6 → 4.0 | 0.56 → 0.96 |
| tripod-far-60 | synth | 47.5 → 89.1 | 3/6 (6) → 6/6 (6) | 1 → 6 | 0.59 → 0.27 | 1.67 → 0.33 | 3.85 → 0.33 | 2.5 → 2.9 | 0.60 → 1.00 |
| tripod-long-30 | synth | 79.3 → 88.5 | 7/8 (7) → 8/8 (8) | 7 → 8 | 0.31 → 0.22 | 1.71 → 0.88 | 0.96 → 0.36 | 4.6 → 4.3 | 0.75 → 0.97 |
| side-pan-raw | limit | 33.9 → 40.0 | 4/6 (4) → 4/6 (4) | 0 → 1 | 5.90 → 3.10 | 0.50 → 0.75 | 6.80 → 6.10 | 4.0 → 4.5 | 0.13 → 0.15 |
| back-row-attack | limit | 44.2 → 75.4 | 3/6 (3) → 5/6 (5) | 2 → 5 | 2.70 → 0.56 | 0.00 → 1.00 | 5.44 → 0.81 | 2.8 → 5.6 | 0.21 → 0.80 |
| long-rally | limit | 63.2 → 86.6 | 7/8 (7) → 8/8 (8) | 5 → 8 | 0.97 → 0.32 | 0.86 → 0.63 | 2.71 → 0.43 | 4.4 → 4.1 | 0.68 → 0.86 |
| pol_ita_r1 | tuning | 60.7 → 78.2 | 3/4 (6) → 4/4 (5) | 3 → 4 | 2.40 → 1.31 | 1.00 → 0.75 | - | 2.5 → 2.2 | 0.72 → 0.64 |
| pol_ita_r2 | tuning | 71.6 → 80.3 | 5/5 (7) → 5/5 (6) | 5 → 5 | 1.89 → 0.97 | 0.60 → 0.60 | - | 4.1 → 3.8 | 0.90 → 0.89 |
| pol_ita_r3 | holdout | 76.7 → 85.8 | 4/4 (6) → 4/4 (5) | 4 → 4 | 1.16 → 0.46 | 1.00 → 0.75 | - | 3.6 → 4.3 | 0.95 → 0.82 |
| pol_ita_r4 | tuning | 78.6 → 78.5 | 4/4 (4) → 4/4 (4) | 3 → 3 | 1.04 → 1.06 | 1.00 → 0.25 | - | 2.2 → 2.1 | 0.84 → 0.84 |
| pol_ita_r5 | tuning | 76.4 → 76.5 | 2/2 (3) → 2/2 (3) | 2 → 2 | 1.06 → 1.06 | 0.50 → 0.50 | - | 5.2 → 5.1 | 0.90 → 0.90 |
| pol_ita_r6 | holdout | 66.2 → 66.1 | 8/8 (10) → 7/8 (8) | 6 → 5 | 2.47 → 1.83 | 1.00 → 0.43 | - | 3.3 → 3.0 | 0.65 → 0.65 |
| pol_ita_r7 | tuning | 59.9 → 65.3 | 4/5 (6) → 4/5 (5) | 4 → 4 | 2.70 → 2.06 | 0.75 → 0.50 | - | 6.7 → 4.8 | 0.42 → 0.30 |
| usa_jap_r1 | tuning | 75.0 → 75.8 | 2/3 (2) → 2/3 (2) | 2 → 2 | 0.92 → 0.85 | 1.00 → 1.00 | - | 1.6 → 1.5 | 1.00 → 1.00 |
| usa_jap_r2 | holdout | 66.1 → 75.9 | 12/14 (15) → 14/14 (15) | 12 → 14 | 2.18 → 1.61 | 1.00 → 0.64 | - | 5.2 → 4.1 | 0.82 → 0.87 |
| usa_jap_r3 | tuning | 55.7 → 72.5 | 11/16 (17) → 13/16 (15) | 8 → 13 | 2.36 → 1.34 | 0.82 → 0.77 | - | 2.7 → 3.0 | 0.91 → 0.82 |
| usa_jap_r4 | tuning | 81.0 → 78.3 | 2/2 (2) → 2/2 (2) | 2 → 2 | 1.39 → 1.65 | 2.00 → 1.50 | - | 1.1 → 1.9 | 0.70 → 0.71 |

Known-limit scenarios are not gated. Two of them are no longer limits (`back-row-attack` 44 → 75, `long-rally` 63 → 87).

## 5. What changed, in the order it was kept

Each is one commit with its metric delta in the message.

1. **Touches are found by scanning for kinks with a flight model that is right under any lens** (`b7972aa`). The old splitter extended an image parabola frame by frame; under a wide lens a flight is not an image parabola, so tripod flights were cut every few frames and the real touches lost. Now every position of the track is scored for "one flight" against "two flights meeting here", using all samples within 0.5 s on both sides and the noise measured on the track itself. Velocity jumps are converted to m/s at the ball's own distance. Rules: the toss is the release from the hand; a ball first seen above reach is not a toss; an attack and its block a few frames apart are placed together. Synthetic 70.6 → 80.4, tuning 69.9 → 73.1, precision 70% → 86%.
2. **Take-off spot of a jumping player** (`8c0a3bf`). It was the lowest feet over ±10 frames, which reaches back into the run-up, when the player was somewhere else. Now a jump is feet that were lower both before and after the touch. Touch position on the tuning set 1.66 → 1.18 m, holdout 2.23 → 1.61 m.
3. **A block follows its attack within 0.4 s; a very clear kink is a touch even with no player detected** (`85a0362`). Tuning type correct 80% → 85%; the hidden-receiver scenario 49 → 83.5.
4. **Editor** (`61b67b8`), section 6.
5. **60 fps** (`2219637`): limits counted in frames mean the same time at 60 fps as at 30, in the analyzer, the ball labeller and the solver. No effect at 25/30 fps.

Tried and not kept: a 3D ray fit for the flight model (collapsed towards the camera on short windows), waiving the velocity test for scanned kinks (false touches where the detector drifts past the net), preferring the player the ball is centred over (no gain once the take-off spot was fixed), re-anchoring touches at refused flights (item 3 above).

No architecture change and no new dependency. vb14 and unlabelled footage were not used for training or tuning.

## 6. Editor

In the Inspect & Edit view's 3D editor (top and side views of the court):

- **Drag a contact point**: as before; it snaps to the camera's line of sight (Shift: free).
- **Add or delete a contact, change its type**: as before, in the Touch panel (`T` adds one at the current frame). New kinds that only the editor sets: free ball, net, floor.
- **Arc handle**: a diamond midway through each flight in the side view. Drag it up or down to make the arc higher or flatter.
- **Re-fit**: only the flights at the edit are rebuilt, at once: one ballistic arc between its two contacts, falling at a rate fitted to that flight's detections within what the contact type allows (a set 0.9-1.15 g, an attack or serve up to 2.2 g), always within 0.3-2.5 g. Flights share their contact points, so the path stays continuous. Rebuilt flights are drawn dashed; a flight no ball could make (over 40 m/s, or through the net) is flagged. The full solve still runs in the background with the same edits.
- **Undo / redo** over every edit (buttons, Ctrl+Z, Ctrl+Shift+Z or Ctrl+Y). **Reset flight to automatic** clears a flight's arc and the placements of its two contacts.
- **Tests** (`src/utils/rallyRefit.test.ts`): untouched flights are unchanged frame for frame; a rebuilt flight is straight over the floor with a constant fall within bounds and never below the floor; it starts and ends exactly on its contacts; every frame between first and last contact has a position; type bounds hold; add, delete, move, arc and reset; latency on a 481-frame, 15-contact rally.

**Not verified:** the re-fit logic is tested, and the app typechecks and builds, but I did not run the app and drag the handles by hand. The drawing and pointer code in `CourtEditor3D.tsx` should be tried once in the real view.

## 7. Performance

Measured with `python python/bench_pipeline.py` (warm: a 2 s clip is analysed first and not counted), 2026-11-13.

| | 10 s rally | 19.2 s rally |
|---|---|---|
| Clip | first 10 s of `usa_jap_r3`, 1920x1080, 25 fps | `usa_jap_r3`, 1920x1080, 25 fps |
| Analysis (ball, players, arms, camera motion, court) | **32.3 s** (3.2 s per s of video) | **54.9 s** (2.9 s per s) |
| 3D solve (TypeScript) | ~1.5 s | 2.9 s (was 5.2 s) |
| Peak VRAM added (nvidia-smi) | 908 MiB | 1047 MiB |
| GPU power, mean / peak | 64 W / 99 W (idle 31 W) | 64 W / 97 W |
| GPU utilisation, mean | 39% | 49% |

- CPU: AMD Ryzen 9 6900HX. GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8 GB. On AC, Windows power scheme "Performance".
- **Caveat:** other applications were holding about 4.7 GB of VRAM and ~30 W during the measurement, so this was not a clean "no other GPU apps" run. The added VRAM is measured against that idle level.
- The Python analyzer was only changed for 60 fps clips, so its speed at 25/30 fps is what it was. It is still CPU-bound (GPU under half used); nothing was spent on GPU speed-ups.
- Not measured: the production RTX 3060 server.

## 8. Main remaining failure cases

- **Blocked balls flying along the line of sight** (item 1 in section 4): the flight after a block is the least certain part of every rally seen from behind.
- **Fast serves and attacks between two anchored touches** are often refused as fits and drawn as estimates (dashed): the path is there and plausible, but it is not measured. This is the coverage shortfall.
- **The wrong player.** On real clips the touch is within 1.2 m at the median, but a touch among several players can still land on a neighbour 2-3 m away along the line of sight (`pol_ita_r7` set and attack). Which player the ball is centred over in the picture is a strong cue (right in 30 of 40 touches offline); wiring it in properly, at the contact frame, is the most promising next step.
- **Jump serves seen from behind the court**: when the server lands further in than they took off, the take-off is not recovered and the serve contact is placed up to ~2.5 m too deep (France vs Japan: 6.9 m from the net instead of ~9.5 m). The rally rules still start correctly.
- **False detections after the ball is lost** (`usa_jap_r4`: the tracker hops to a player's shorts and then the floor). No longer reported as a touch, but the path follows them.
- **Touches the detector never sees**: `usa_jap_r3` f218/f222 (attack and block), f304 (dig); `pol_ita_r7` f128 (block 2 frames after the attack).
- **An overpass attacked directly** (`pol_ita_r4` f132): labelled attack, found as "set", because the sequence expects a set after the reception.

## 9. What is still weak

- **No real tripod footage.** The tripod numbers are synthetic: 85 mean score, 30 of 32 touches typed right. The synthetic players jump on the spot and the detections have plain pixel noise; real phone footage will have motion blur at 30 fps, a ball detector trained on broadcast footage, and players moving through each other. Expect the first real clips to be worse than this, mostly in detection.
- **Line coverage on real clips is 93% on one clip** (`usa_jap_r4`), not 100%: its first labelled contact is before the first detection of the ball, and nothing is drawn before the ball is first seen. From the first frame the ball is seen to the last contact, every clip has a path (tested).
- **Floor contacts, net touches and free balls are not found automatically**; they can be set in the editor.
- **11 clips from 2 matches.** The 95% interval on the real-clip score is about ±4 points, and the tuning-set type and precision criteria are met by one touch each (35/41, 36/42).
- **Ball size** is only used to choose a scale for velocity tests and as a soft prior, not to place the ball.

## 10. Labels that look wrong (not changed)

- `usa_jap_r1`: two toss labels on consecutive frames (f26 and f27). One of them can never be matched, which costs that clip a third of its recall.

I also checked two "false" receptions after serves against the video, in case they were real touches missing from the labels. `usa_jap_r4` f57 is not one: the detector is on a player and then the floor. `pol_ita_r5` f69 I did not settle.

## 11. Recording guide (tripod footage)

**Position**
- Behind the end line, on the court's long axis (within ~1 m of centre), **5-8 m back** and **3-4 m high**. Higher is better: from 3 m a far player's jump shifts their feet in the picture as much as about 4 m of court does; from 4 m, about a quarter less.
- Landscape, main lens (not ultra-wide: the ball should stay ≥ 9 px at the far baseline). All four corners of the court in the picture, and at least 5 m of air above the net.
- If the near server ends up out of the picture, accept that rather than switch to ultra-wide.

**Camera**
- 1080p. **60 fps if the light allows**, else 30. Record some rallies at each: 60 fps is what separates an attack from its block.
- Lock exposure and focus (long-press on the court); a shutter of 1/500 s or faster if the camera app lets you set it. Motion blur is what the detector fails on.
- Stabilisation **off**, tripod weighted or taped. The camera must not move during a rally; if it is bumped, start a new file.
- No zoom changes, no HDR video, no variable frame rate.

**What to record**
- One file per rally if convenient, or long takes plus a list of start times. Start ~2 s before the serve toss and stop ~2 s after the ball is dead.
- At least 20 rallies, from both ends if possible, including jump serves, tips, block-outs, long rallies and balls into the net.
- A ball that contrasts with the walls and floor; avoid a banner in the ball's colour directly behind play.
- One still of the empty court from the same position, for the court calibration.

**Then**
1. `python python/export_fixture.py clip.mp4` (30 and 60 fps both work; a static camera needs no camera-motion data).
2. `python python/label_touches.py clip.mp4` for contacts and feet, about 2 minutes a rally.
3. `python python/label_ball.py clip.mp4`, on the frames it jumps to with `g` (where the detector lost the ball). A few hundred frames across clips is enough to start retraining with `train_ball.py`.
4. `npm run eval`.

Checked for this: `export_fixture.py` on a static 30 fps clip reproduces the committed fixture's 139 detections, and a 60 fps clip exports and solves. The 60 fps clip was a 30 fps one with every frame doubled, so it tests the code path, not detection at 60 fps. `label_ball.py` is interactive and was not run; its frame-rate handling was changed and compiles.

## 12. Commands

| Command | |
|---|---|
| `npm run eval` | all cases; prints the three group lines these tables are made from |
| `npm run eval:diag -- clip[,clip]` | one clip: each true touch against the found one, path error per flight, the solver's verdict on every segment |
| `npm run eval:splits` | every candidate touch split with its evidence, on the synthetic and tuning clips |
| `npm test` | unit tests, per-case regression gate (raised to the current scores), acceptance tests, coverage floor, editor re-fit tests |
| `python python/bench_pipeline.py clip.mp4 --seconds 10` | analysis time, VRAM, GPU power |
