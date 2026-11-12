# Ball-reconstruction eval

Scores the 3D rally solver against ground truth, so changes are measured instead of judged by eye.

| Command | What it does |
|---|---|
| `npm run eval` | Runs every synthetic scenario and real fixture, prints a score table, writes `eval/report.html` |
| `npm run eval -- --only near-serve,France_vs_Japan` | Just those cases |
| `npm run eval -- --params eval/tuned-params.json` | With parameter overrides |
| `npm run eval -- --save-baseline` | Stores current scores as the regression baseline (`eval/baseline.json`) |
| `npm test` | Unit tests, a reviewed check on France vs Japan, and the regression gate (no case may drop >3 points below baseline) |
| `npm run tune -- --iters 40 --sensitivity` | Searches solver parameters; writes `eval/tuned-params.json` and `eval/tune-report.json` |

## Adding a real clip
Easiest: `python python/clip_picker.py` opens a page at http://127.0.0.1:8765. Search YouTube or pick a suggested match, mark the rally (I = start, O = end, P = preview) and press Download clip. Only that section is downloaded (1080p, video only, to `eval/videos/`), the analyzer writes the fixture, and the Label button opens the labeller. Downloads are recorded in `eval/clip_sources.json`, so `python python/fetch_clip.py --list eval/clip_sources.json` rebuilds the set elsewhere.

By hand:
1. `python python/fetch_clip.py URL 1:02:10-1:02:45 --name usa_bra_r1 --fixture`, or for a local file `python python/export_fixture.py clip.mp4` → `eval/fixtures/clip.json` (runs the analyzer once, ~40 s on GPU).
2. `python python/label_touches.py clip.mp4` → `eval/labels/clip.json`. Mark each touch at its contact frame and click the toucher's feet (take-off spot for jumps). About 1–2 minutes per rally.
3. Note the clip's source in `clips.md`. Videos are not committed; fixtures and labels are.
4. `npm run eval`, then `npm run eval -- --save-baseline` once the scores are reviewed.

## Seeing where a clip goes wrong
`npm run inspect` opens http://127.0.0.1:8766: pick a clip (real or synthetic) and the solver runs on it, drawn over the video stage by stage: detections, camera motion, the fitted court, the solved ball reprojected with its floor shadow, arcs, touch splits and touches against the labels. A list of likely problems names the stage each comes from (e.g. "split rejected, Δv 2.1 < touchMinDvMps 3.55" or "ball not detected for 27 frames"); click one to jump there. Every Run rebuilds the solver, so code edits show without restarting. Parameter overrides can be tried on one clip, or on all clips against the baseline.

## Estimated stretches of the line
The rally solver refuses flights it cannot fit, often fast serves and attacks with most of their detections present. So the line still runs through the rally, `fillLineGaps` fills those frames after the solve, with method `estimated`, drawn dashed in the 3D view:
- A refused flight keeps its solved arc, which shares knots with its neighbours, moved onto the detections' lines of sight (depth from the arc, the rest from the video).
- A hole of up to 2.5 s between flights gets a ballistic arc, unless the ball was on the floor or the next thing is a toss or serve.
- Otherwise the last flight is followed under gravity for up to 1.5 s or until it reaches the floor.

Estimates are shapes, not measurements. Touches are decided before the fill, and point winners and the inspector's problems read fitted frames only. Synthetic path metrics do include them. Where the camera could not see the ball at all, nothing but the arc says where it was.

## Choosing the ball detector
The analyzer runs one of three multi-frame networks, picked with `BALL_DETECTOR` (weights download on first use, pinned and checked by SHA-256). Best real-clip score on replay, with motion stabilisation:

| `BALL_DETECTOR` | Network | Frames per window | Weights | Score |
|---|---|---|---|---|
| `tracknetv2` (default) | TrackNetV2 | 3 RGB | [WASB-SBDT](https://github.com/nttcom/WASB-SBDT) volleyball, broadcast indoor | 71 |
| `vballnet` | VballNetV4c (ONNX, CPU) | 9 grayscale | [asigatchov/fast-volleyball-tracking-inference](https://github.com/asigatchov/fast-volleyball-tracking-inference) | 55 |
| `tracknetv3` | TrackNetV3 | 8 RGB + the clip's median frame | [deadfast/beach-volley-vision-models](https://huggingface.co/deadfast/beach-volley-vision-models), beach | 28 |

Windows are warped to cancel camera motion before detection (`BALL_STABILIZE=0` turns that off): the networks read the ball from what changes between frames, and on a panning camera everything does. `BALL_THRESHOLD` overrides the heatmap threshold (0.5) and `BALL_STRIDE` how far a window advances. `train_ball.py` fine-tunes TrackNetV2 only.

A full export takes a minute or more a clip, nearly all of it on things a ball-detector change doesn't affect. `python/bench_ball.py` caches each network's heatmaps once and rebuilds fixtures from them in seconds, for any threshold, tracker setting or mix of networks:
```
python python/bench_ball.py cache --detector vballnet+stab --stride 3
python python/bench_ball.py replay --detector vballnet+stab --stride 3 --threshold 0.5 --out runs/fx_vball
EVAL_FIXTURES_DIR=runs/fx_vball npm run eval -- --real
```
Check a final choice with real exports (`export_fixture.py --out-dir runs/fx`): replay came out about a point above them.

## What the score does and doesn't tell you
- **Read the real-clip line**, `Real clips: 69.8  95% CI 64.8-74.6`. The "Overall" line above it also averages the synthetic scenarios, which a detector change can't move. The CI is about ±5 because there are only 11 clips from 2 matches, so compare options clip by clip rather than by their means.
- **Which court.** By default real clips are solved on the labelled court, which isolates ball tracking. `EVAL_COURT=detected` uses the analyzer's court, as the app does; the two should agree. Truth positions always come from the labelled court.
- **The score barely sees small court errors**: halving the court error from 14 to 7 px moved it by under a point. Judge the court by `court error` and the `on-line share` the analyzer logs (share of the court model on lines that persist across the clip), not by the score.
- **Labelled courts aren't exact.** Those copied with `propagate_court.py` are up to ~45 px off at the far end of a clip's view, and the analyzer's court sits on the painted lines better than the label on all 11 clips. A large `court error` on one clip can be the label's.
- `npm run eval:audit` checks the score itself: what a pipeline with no information scores (22) and how it falls with fewer, noisier or false detections. It falls steadily on all three, and the solver is deterministic, so differences on the same fixtures are real. No ball positions are labelled, so detector accuracy is only measured through the touches it leads to; `python/label_ball.py` would give a direct measure.

## Retraining the ball detector
TrackNet's published weights miss fast, motion-blurred balls on this footage (a spike is a streak several ball widths long per frame). Retrain it on your own labels:
1. `python python/label_ball.py eval/videos/clip.mp4` → `eval/ball_labels/clip.json`. Suggestions are pre-marked (yellow: the analyzer's ball, magenta 1-9: moving yellow blobs), so most frames take one key: Enter accepts yellow, a number accepts a magenta candidate, a click places the ball, `n` means no ball visible. `g` jumps to the next frame where the detector lost the ball during play; those matter most. Label runs of consecutive frames, and include some easy and some no-ball frames. A few hundred frames across several clips is a useful start.
2. `python python/train_ball.py` fine-tunes from the published weights. The last 20% of every labelled clip is held out; both models are scored on it (precision / recall / F1) and the new weights are saved only if better, to `python/models/tracknetv2_volleyball_finetuned.pth.tar`. The analyzer uses that file when it exists; delete it to go back. `--eval-only` just scores the current weights.
3. Re-export the fixtures (`export_fixture.py`) and run `npm run eval` to see the effect on the 3D results.

`EVAL_FIXTURES_DIR=other/folder npm run eval` scores fixtures from another folder, for trying an analyzer change before replacing the real ones.

## Side views and moving cameras
- The analyzer measures pan/zoom per frame (`python/camera_motion.py`) and stores the ball, its size and the players in the court-calibration frame's coordinates (`ball_ref_pos`, `ball_ref_diameter_px`, `players_ref`, `ref_from_frame`); the 3D solver uses those automatically. Static clips are unchanged.
- Check the motion estimate: `python python/debug_camera.py eval/videos/clip.mp4` writes a stabilised video with the court drawn on it; the court must stand still.
- Automatic court detection (`python/court_lines.py`) fits the painted lines, so it handles side views and baselines out of shot; `python python/bench_court_lines.py` scores it against the hand-placed courts below. To place a court by hand, in `label_touches.py` press `k` for court mode, `v` until the view matches (side / behind), then pick a point with its number key and click it. Corners are not required: where the 3 m lines and the centre line meet the outer lines work too, and each point can be placed on a different frame while the camera pans. From 4 points the fitted court is drawn on every frame; check it sits on the real lines, then `w` to save.

## Folders
- `fixtures/` analyzer output per clip (tracked)
- `labels/` human touch labels per clip (tracked)
- `baseline.json` regression baseline (tracked)
- synthetic scenarios live in `src/eval/scenarios.ts`; known-limit scenarios are reported but not gated
