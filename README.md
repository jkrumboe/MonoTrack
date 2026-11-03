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
