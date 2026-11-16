# Volleyball Analyzer

Veo for volleyball. Put a phone on a tripod, record the match, and get every
rally, point and touch on a timeline: serve, reception, set, attack, block, dig.

## How it fits together

```
 phone (web app / PWA, or Tauri mobile)      browser (web app)          desktop (Tauri)
   record ─► IndexedDB ─► chunked upload        upload a video            open a local file
                     │                              │                          │
                     ▼                              ▼                          ▼
             python/server.py  (job queue, match library, video streaming)   src-tauri (local)
                     └──────────────► python/analyzer.py ◄─────────────────────┘
                                  court, ball (TrackNet), players, pose
                                               │  frame messages
                                               ▼
              in the browser: camera calibration ─► 3D rally solver (touches)
                              ─► utils/matchTimeline.ts (rallies, points, score)
```

- **One UI, three platforms.** `src/platform/backend.ts` picks a backend:
  the desktop app analyses on the machine (`tauriBackend.ts`); a browser or
  phone talks to the analysis server (`serverBackend.ts`).
- **Match timeline** (`src/utils/matchTimeline.ts`). A rally starts at a
  serve. Volleyball uses rally scoring, so the point goes to whoever serves
  next. When there is no next serve, it goes by where the ball landed
  (in: against that half; out: against the last toucher). Points can be set
  by hand in the Match tab. They are saved with the video's edits, along with
  team names.
- **Recording** (`src/components/RecordView.tsx`). Uses the back camera at
  1080p (30 or 60 fps). A framing guide and a level help set up the tripod,
  and a wake lock keeps the screen on. Every 4 s of video is written to
  IndexedDB, so a crash loses only seconds. Uploads resume after a dropped
  connection.

## Running it

Python setup (once):

```sh
python -m venv .venv && .venv/Scripts/pip install -r python/requirements.txt
```

**Desktop app** (analyses on this machine):

```sh
npm run tauri dev
```

**Web app + analysis server:**

```sh
npm run build
.venv/Scripts/python python/server.py   # serves the app and the API on :8765
```

For development, run `npm run server` and `npm run web` (Vite on :1420,
which proxies `/api` to the server). Settings:

- `VB_DATA`: where matches are kept. The default is `~/.tennis-3d/server`.
- `VB_TOKEN`: requires a token on every API call. Enter the token on the home
  screen under *Analysis server*.
- ffmpeg on PATH: phone recordings (variable frame rate, often WebM) are
  re-encoded to constant-rate MP4.

**Analysis server in Docker (GPU):** on a machine with an NVIDIA GPU, the
driver and the NVIDIA Container Toolkit (Docker Desktop on Windows: WSL 2
backend):

```sh
docker compose up -d --build        # http://<host>:8765, app and API
docker compose logs -f              # analyzer output
```

Matches, saved courts and downloaded weights live in Docker volumes and
survive `docker compose down`. Set `VB_TOKEN` (environment or `.env`) to
require a token. To run the analyzer code from your checkout instead of the
copy in the image (edit, then `restart`, no rebuild):

```sh
docker compose -f compose.yaml -f compose.dev.yaml up -d
```

With Portainer (a standalone Docker environment, not Swarm; the host needs
the NVIDIA Container Toolkit): *Stacks → Add stack → Repository*, this
repository's URL, the branch as reference (`refs/heads/<branch>`), compose
path `compose.yaml`, and `VB_TOKEN` under *Environment variables*. Each
*Pull and redeploy* (or GitOps update) rebuilds from the pulled code.

Using it from a development machine, set `VB_ANALYSIS_SERVER`:

```sh
VB_ANALYSIS_SERVER=http://gpu-box:8765 npm run web             # Vite proxies /api there
VB_ANALYSIS_SERVER=http://gpu-box:8765 python python/export_fixture.py clip.mp4
```

`export_fixture.py` (and `fetch_clip.py --fixture` and `clip_picker.py`,
which run it) then uploads the clip, analyses it on the server's GPU and
writes the eval fixture locally; this machine needs no torch or CUDA.
`npm test` and `npm run eval` read those fixtures and need no GPU.

**Checking a GPU server** in about two minutes (a 20 s stretch of a video,
not the whole match):

```sh
python python/server_check.py --server http://gpu-box:8765 --gpu-only      # 5 s: GPU visible to torch?
python python/server_check.py --server http://gpu-box:8765 --expect-gpu 3060
python python/server_check.py eval/videos/vb14/*.avi --seconds 10          # several videos
```

It reports PASS/WARN/FAIL (torch sees CUDA, GPU busy during the analysis,
frames, ball, court), the speed in seconds per second of video, and what the
whole video would take. Results go to `runs/server_check.jsonl`.

**Phone:** browsers only allow the camera over **https** (or on localhost).
Start the server with a certificate, open its address on the phone, then use
*Add to Home Screen*:

```sh
python python/server.py --cert cert.pem --key key.pem
```

A self-signed certificate works on a LAN once the phone trusts it. Tools like
`mkcert`, or a tunnel/reverse proxy with a real certificate, are easier.

**Native mobile app (Tauri 2):** the same frontend runs as an Android/iOS app,
which uses the server backend automatically:

```sh
npm run tauri android init     # needs Android SDK + NDK
npm run tauri ios init         # needs Xcode (macOS)
```

After `init`, set up the generated projects:

- Android: add the `CAMERA` permission to `AndroidManifest.xml`. Use https
  for the server, or allow cleartext for your server's address.
- iOS: add `NSCameraUsageDescription` and `NSMotionUsageDescription` to
  `Info.plist`.
- Set the server address in the app under *Analysis server*, or build with
  `VITE_API_URL=https://your-server`.

## Still to do for a full product

- **Full-match length.** The analyzer and the in-browser solver were built
  for single-rally clips. An hour of video is ~100k frames, which is too many
  to hold and solve in one go. They should be processed in rally-sized
  windows: find active play first (ball/motion), then run the heavy models
  only there.
- **Calibrating without the user.** Rallies and points appear only after the
  court is calibrated. The auto court detector must work reliably from the
  tripod viewpoint (behind an end line, elevated) so no one has to align the
  court by hand.
- **Set and side changes.** Teams swap ends between sets. The timeline counts
  by court side ("near"/"far"), so a set boundary needs to swap the team
  names.
- **Hosting.** The server has a single-user token, local disk and one GPU
  worker. A public service needs accounts, object storage for videos, a job
  queue with GPU workers, and sharing links for clips.
- **Highlights export.** Cutting the rallies or chosen touches into a reel
  with ffmpeg on the server.
