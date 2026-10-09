#!/usr/bin/env python3
"""
Download only the parts of a YouTube video you need, as eval clips.

    python python/fetch_clip.py URL 1:02:10-1:02:45 --name usa_bra_r1
    python python/fetch_clip.py URL 12:00-12:30 25:40-26:05 --name pol_ita   # two clips
    python python/fetch_clip.py --list eval/clip_sources.json                 # every clip in the list
    add --fixture to also run the analyzer and write eval/fixtures/<name>.json

Times are [h:]mm:ss (or plain seconds). Only the requested sections are
downloaded, video only (no audio), at most 1080p, cut on exact frames.
Clips go to eval/videos/ (not committed; see .gitignore). Each download is
recorded in eval/clip_sources.json so the set can be rebuilt on another
machine with --list.

Use footage you are allowed to use; the clips stay local for testing.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.normpath(os.path.join(HERE, os.pardir, "eval"))
VIDEO_DIR = os.path.join(EVAL_DIR, "videos")
SOURCES = os.path.join(EVAL_DIR, "clip_sources.json")


def parse_time(s):
    """'1:02:10' / '62:10' / '3730.5' -> seconds."""
    parts = [float(p) for p in s.strip().split(":")]
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs


def parse_range(r):
    a, b = r.split("-", 1)
    start, end = parse_time(a), parse_time(b)
    if end <= start:
        raise ValueError(f"range {r!r}: end must be after start")
    return start, end


def fmt(secs):
    h, rem = divmod(int(round(secs)), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def download(url, start, end, name, max_height=1080):
    """Fetch [start, end] seconds of `url` to eval/videos/<name>.mp4."""
    import yt_dlp
    from yt_dlp.utils import download_range_func

    os.makedirs(VIDEO_DIR, exist_ok=True)
    out = os.path.join(VIDEO_DIR, name + ".mp4")
    opts = {
        # Video only: the analyzer never uses audio, and it halves the size.
        "format": f"bv*[height<={max_height}][ext=mp4]/bv*[height<={max_height}]/b[height<={max_height}]",
        "outtmpl": os.path.join(VIDEO_DIR, name + ".%(ext)s"),
        "download_ranges": download_range_func(None, [(start, end)]),
        # Re-encode at the cut points so the clip starts on the exact frame
        # instead of the previous keyframe (which can be seconds earlier).
        "force_keyframes_at_cuts": True,
        "merge_output_format": "mp4",
        "remux_video": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "overwrites": True,
    }
    print(f"[fetch] {name}: {fmt(start)}-{fmt(end)} ({end - start:.0f} s) from {url}")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    if not os.path.isfile(out):
        sys.exit(f"[fetch] download finished but {out} is missing")
    size_mb = os.path.getsize(out) / 1e6
    print(f"[fetch] -> {os.path.relpath(out)} ({size_mb:.1f} MB, {info.get('height', '?')}p)")
    return out, info


def record(entry):
    """Add or replace an entry in eval/clip_sources.json."""
    entries = []
    if os.path.isfile(SOURCES):
        with open(SOURCES, encoding="utf-8") as fh:
            entries = json.load(fh)
    entries = [e for e in entries if e["name"] != entry["name"]] + [entry]
    entries.sort(key=lambda e: e["name"])
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(SOURCES, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2)


def make_fixture(path, name):
    subprocess.run([sys.executable, os.path.join(HERE, "export_fixture.py"), path, "--name", name], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", nargs="?", help="YouTube URL")
    ap.add_argument("ranges", nargs="*", help="start-end, e.g. 1:02:10-1:02:45")
    ap.add_argument("--name", help="clip name (several ranges get _1, _2 ...)")
    ap.add_argument("--note", default="", help="what the clip shows (stored in clip_sources.json)")
    ap.add_argument("--list", help="download every clip in a clip_sources.json-style file")
    ap.add_argument("--fixture", action="store_true", help="also run the analyzer and write the eval fixture")
    ap.add_argument("--max-height", type=int, default=1080)
    args = ap.parse_args()

    jobs = []
    if args.list:
        with open(args.list, encoding="utf-8") as fh:
            for e in json.load(fh):
                jobs.append((e["url"], parse_time(e["start"]), parse_time(e["end"]), e["name"], e.get("note", "")))
    else:
        if not args.url or not args.ranges:
            ap.error("give a URL and at least one start-end range (or --list)")
        base = args.name or "clip"
        for i, r in enumerate(args.ranges):
            start, end = parse_range(r)
            name = base if len(args.ranges) == 1 else f"{base}_{i + 1}"
            jobs.append((args.url, start, end, name, args.note))

    for url, start, end, name, note in jobs:
        path, info = download(url, start, end, name, args.max_height)
        record({
            "name": name, "url": url, "start": fmt(start), "end": fmt(end),
            "title": info.get("title", ""), "channel": info.get("channel", ""),
            "height": info.get("height"), "fps": info.get("fps"), "note": note,
        })
        if args.fixture:
            make_fixture(path, name)


if __name__ == "__main__":
    main()
