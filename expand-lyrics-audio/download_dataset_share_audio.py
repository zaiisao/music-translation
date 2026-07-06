"""Download audio for dataset_share/meta.csv rows that have a YouTube/SoundCloud URL.

Usage:
    python download_dataset_share_audio.py [--limit N] [--start-lid LID]

Writes audio to dataset_share/audio/<LID>.<ext> and a per-run log to
dataset_share/download_log.csv (lid, status, url, detail). Safe to re-run:
rows whose audio file already exists are skipped.
"""
import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
META_CSV = ROOT / "dataset_share" / "meta.csv"
AUDIO_DIR = ROOT / "dataset_share" / "audio"
LOG_CSV = ROOT / "dataset_share" / "download_log.csv"

URL_PATTERN = re.compile(r"(youtu\.be|youtube\.com|soundcloud\.com)")


def pick_url(row):
    for key in ("url1", "url2"):
        u = (row.get(key) or "").strip()
        if u and URL_PATTERN.search(u):
            return u
    return None


def already_downloaded(lid):
    return any(AUDIO_DIR.glob(f"{lid}.*"))


def download_one(lid, url):
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(AUDIO_DIR / f"{lid}.%(ext)s")
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "best",
        "--no-playlist",
        "--no-warnings",
        "--sleep-interval", "1",
        "--max-sleep-interval", "3",
        "-o", out_tmpl,
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode == 0 and already_downloaded(lid):
        return "ok", ""
    detail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return "failed", detail[-1] if detail else f"exit {proc.returncode}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start-lid", default=None, help="resume from this LID (inclusive)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(META_CSV, newline="", encoding="utf-8")))
    if args.start_lid:
        idx = next((i for i, r in enumerate(rows) if r["LID"] == args.start_lid), 0)
        rows = rows[idx:]
    if args.limit:
        rows = rows[: args.limit]

    log_is_new = not LOG_CSV.exists()
    log_f = open(LOG_CSV, "a", newline="", encoding="utf-8")
    writer = csv.writer(log_f)
    if log_is_new:
        writer.writerow(["lid", "status", "url", "detail"])

    counts = {"ok": 0, "skipped_exists": 0, "skipped_no_url": 0, "failed": 0}
    for row in rows:
        lid = row["LID"]
        url = pick_url(row)
        if not url:
            counts["skipped_no_url"] += 1
            continue
        if already_downloaded(lid):
            counts["skipped_exists"] += 1
            continue
        status, detail = download_one(lid, url)
        counts[status] = counts.get(status, 0) + 1
        writer.writerow([lid, status, url, detail])
        log_f.flush()
        print(f"{lid}: {status} {detail}", file=sys.stderr)

    log_f.close()
    print(counts)


if __name__ == "__main__":
    main()
