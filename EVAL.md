# Rally pipeline: eval log

Status: **in progress**. This file is updated after every kept change.

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

## 3. Hardware

- CPU: AMD Ryzen 9 6900HX
- GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8 GB
