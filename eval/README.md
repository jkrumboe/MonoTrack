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
