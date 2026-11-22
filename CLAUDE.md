# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Despite the repo name, this is a **volleyball** analyzer ("Veo for volleyball"): a video of a match goes in, and rallies, points and touches (serve, reception, set, attack, block, dig) come out on a timeline. `README.md` covers setup and deployment in detail. `eval/README.md` covers the eval tooling. `EVAL.md` is the running log of accuracy work.

## Commands

```sh
npm run typecheck            # tsc on the app and on the eval code (tsconfig.eval.json)
npm test                     # vitest: unit tests + the eval regression gate (slow, runs the full solver)
npx vitest run src/utils/matchTimeline.test.ts      # one test file
npx vitest run -t "camera roll"                     # tests by name

npm run eval                                  # score the solver on every synthetic scenario and real fixture
npm run eval -- --only near-serve,France_vs_Japan
npm run eval -- --real                        # real clips only
npm run eval -- --save-baseline               # only after the scores have been reviewed
npm run eval:diag -- <clip>                   # where one clip goes wrong, as text
npm run inspect                               # visual solver debugger on http://127.0.0.1:8766

npm run server               # python/server.py, analysis server + API on :8765
npm run web                  # Vite on :1420, proxies /api to the server (or to $VB_ANALYSIS_SERVER)
npm run tauri dev            # desktop app, runs the analyzer locally
```

Python lives in `.venv` (`.venv/Scripts/python` on Windows). The Python side has no test suite. Its changes are checked by re-exporting fixtures (`python/export_fixture.py`), then running `npm run eval`, and by timing the `python/bench_*.py` scripts (`bench_pipeline.py` for the whole analyzer warm, plus ball, court_lines and players).

## Architecture

The work is split across two runtimes:

1. **Python analyzer** (`python/analyzer.py`): the CV half. It runs court detection (`court_lines.py`, `court_detect.py`), the ball (one of the TrackNet-family networks, picked with `BALL_DETECTOR`; see `ball_tracknet.py` and `ball_vballnet.py`), players (YOLO), pose, and camera pan/zoom (`camera_motion.py`). It is a sidecar process that speaks **JSON lines on stdin/stdout**: per-frame messages, then progress, then `complete`. `--warm` keeps it running between videos. Its message format is the contract with the front end (`src/hooks/useAnalysis.ts` types, `src/platform/backend.ts`).
2. **TypeScript solver, in the browser**: camera calibration (`src/utils/cameraSolver.ts`, `courtHomography.ts`), then the 3D rally solver (`src/utils/ballReconstruction.ts`, run in a Web Worker via `solverWorker.ts`) that fits ball flights and finds touches, then `src/utils/matchTimeline.ts` (rallies, rally-scoring points, score).

Who runs the analyzer depends on the platform. `src/platform/backend.ts` picks one of two backends:
- `tauriBackend.ts`: the desktop app. Rust (`src-tauri/src/lib.rs`) spawns `python/analyzer.py` with the venv Python and relays its messages.
- `serverBackend.ts`: browser, PWA and Tauri mobile. Videos are uploaded in chunks to `python/server.py`, which does the same job as `lib.rs` (one warm analyzer, a job queue) and also stores the match library. `server.py` uses the **standard library only**.

On a moving camera, the analyzer stores the ball and players in the court-calibration frame's coordinates (`ball_ref_pos`, `players_ref`, `ref_from_frame`), and the solver uses those when they are present.

## Eval harness (`src/eval/`, `eval/`)

- Cases are the synthetic scenarios in `src/eval/scenarios.ts` plus the real clips: `eval/fixtures/*.json` (analyzer output) and `eval/labels/*.json` (hand-labelled touches). Fixtures and labels are committed. Videos are not.
- `HOLDOUT` in `src/eval/dataset.ts` lists the 3 holdout clips. **Never use them to choose parameters or rules.** Tune on the 8 tuning clips and report holdout separately.
- `npm test` includes a regression gate: no case may drop more than 3 points below `eval/baseline.json`. It also checks the acceptance criteria from `EVAL.md` and a reprojection-coverage floor. A change that lowers scores on purpose needs a reviewed `--save-baseline`.
- When comparing scores, read the **real-clip line**, not "Overall", because the synthetic scenarios can't move with a detector change. With 11 clips the 95% CI is about ±5, so compare clip by clip.
- `EVAL_FIXTURES_DIR=<dir>` scores fixtures from another folder. Use it to try an analyzer change without overwriting `eval/fixtures/`. `EVAL_COURT=detected` uses the analyzer's court instead of the labelled one.
- Record results, and things tried and not kept, in `EVAL.md` in its existing style.

## Conventions

- Commit messages are `area: summary` in lowercase, with the summary in plain words (e.g. `court: line mask in 8-bit integers`, `players: YOLOv8s on every 2nd frame`).
- Model weights are pinned and SHA-256 checked, and download on first use into `python/models/`. A fine-tuned `python/models/tracknetv2_volleyball_finetuned.pth.tar` is used automatically when it exists.
- Frame-count limits scale with the frame rate above 45 fps (60 fps recordings). Keep that in mind when adding thresholds in frames.
