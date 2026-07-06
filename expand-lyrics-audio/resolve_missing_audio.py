"""Resolve a YouTube candidate for dataset_share/meta.csv rows that have no
usable audio URL (no youtu.be/youtube.com/soundcloud link in url1/url2).

Does NOT download anything -- it only searches (yt-dlp ytsearch, flat-playlist,
no media fetch) and scores candidates, writing everything to a review CSV so a
human can spot-check before any download is attempted.

Usage:
    python resolve_missing_audio.py [--limit N]

Output: dataset_share/audio_resolution.csv
    lid, artist, english, korean, genre, query, decision, score,
    best_title, best_channel, best_duration, best_url, alt1_title, alt1_url, alt2_title, alt2_url
"""
import argparse
import csv
import difflib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
META_CSV = ROOT / "dataset_share" / "meta.csv"
OUT_CSV = ROOT / "dataset_share" / "audio_resolution.csv"

AUDIO_URL_PATTERN = re.compile(r"(youtu\.be|youtube\.com|soundcloud\.com)")
BLACKLIST = re.compile(
    r"\b(reaction|dance practice|practice video|cover by|tutorial|teaser|trailer|"
    r"shorts|8d audio|sped up|slowed|nightcore|compilation|full album|"
    r"1 hour|10 hours|karaoke|instrumental|mv reaction)\b",
    re.IGNORECASE,
)
# K-pop artists frequently upload English/Japanese/Chinese re-recordings from the
# same official channel; the title/channel heuristics alone can't tell those apart
# from the Korean original, so penalize explicit non-Korean version tags directly.
NON_KOREAN_VERSION = re.compile(
    r"\((?:eng(?:lish)?|jp|japanese|chinese|mandarin)[\s.]*(?:ver(?:sion)?)\.?\)"
    r"|\b(?:english|japanese|chinese|mandarin)\s+ver(?:sion)?\b",
    re.IGNORECASE,
)
N_CANDIDATES = 5
MIN_DURATION = 45
MAX_DURATION_FULL_CREDIT = 480
MAX_DURATION_PARTIAL = 600

AUTO_ACCEPT_THRESHOLD = 0.75
NEEDS_REVIEW_THRESHOLD = 0.45


def normalize(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def has_no_audio_url(row):
    u1, u2 = (row.get("url1") or "").strip(), (row.get("url2") or "").strip()
    return not any(u and AUDIO_URL_PATTERN.search(u) for u in (u1, u2))


def build_query(row):
    artist = (row.get("Artist") or "").strip()
    title = (row.get("English") or "").strip() or (row.get("Korean") or "").strip()
    return f"{artist} {title} official audio".strip()


def channel_score(channel, artist):
    a = normalize(artist)
    c = normalize(channel)
    if not a or not c:
        return 0.0
    a_squashed, c_squashed = a.replace(" ", ""), c.replace(" ", "")
    if a == c or a_squashed == c_squashed:
        return 1.0
    if a in c or c in a or a_squashed in c_squashed or c_squashed in a_squashed:
        return 0.9
    if re.search(r"- ?topic$", (channel or "").strip(), re.IGNORECASE):
        return 0.85
    return difflib.SequenceMatcher(None, a, c).ratio() * 0.5


def title_score(candidate_title, artist, title):
    expected = normalize(f"{artist} {title}")
    cand = normalize(candidate_title)
    if not expected or not cand:
        return 0.0
    return difflib.SequenceMatcher(None, expected, cand).ratio()


def duration_score(duration):
    if duration is None:
        return 0.5
    if duration < MIN_DURATION:
        return 0.0
    if duration <= MAX_DURATION_FULL_CREDIT:
        return 1.0
    if duration <= MAX_DURATION_PARTIAL:
        return 0.5
    return 0.0


def score_candidate(cand, artist, title, genre):
    cand_title = cand.get("title") or ""
    if BLACKLIST.search(cand_title):
        return -1.0
    ts = title_score(cand_title, artist, title)
    cs = channel_score(cand.get("channel") or cand.get("uploader") or "", artist)
    ds = duration_score(cand.get("duration"))
    score = 0.45 * ts + 0.35 * cs + 0.20 * ds
    if genre == "k-pop" and NON_KOREAN_VERSION.search(cand_title):
        score -= 0.35
    return score


def search(query):
    cmd = [
        "yt-dlp", f"ytsearch{N_CANDIDATES}:{query}",
        "--dump-json", "--flat-playlist", "--skip-download", "--no-warnings",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return []
    candidates = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        candidates.append(d)
    return candidates


def decide(score):
    if score >= AUTO_ACCEPT_THRESHOLD:
        return "auto-accept"
    if score >= NEEDS_REVIEW_THRESHOLD:
        return "needs-review"
    return "no-match"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(META_CSV, newline="", encoding="utf-8")) if has_no_audio_url(r)]
    if args.limit:
        rows = rows[: args.limit]

    out_f = open(OUT_CSV, "w", newline="", encoding="utf-8")
    writer = csv.writer(out_f)
    writer.writerow([
        "lid", "artist", "english", "korean", "genre", "query", "decision", "score",
        "best_title", "best_channel", "best_duration", "best_url",
        "alt1_title", "alt1_url", "alt2_title", "alt2_url",
    ])

    counts = {"auto-accept": 0, "needs-review": 0, "no-match": 0}
    for i, row in enumerate(rows):
        artist = row.get("Artist") or ""
        title = (row.get("English") or "").strip() or (row.get("Korean") or "").strip()
        query = build_query(row)
        candidates = search(query)
        genre = row.get("Genre") or ""
        scored = sorted(
            ((score_candidate(c, artist, title, genre), c) for c in candidates),
            key=lambda x: x[0], reverse=True,
        )
        if not scored or scored[0][0] <= 0:
            decision, best = "no-match", None
        else:
            decision = decide(scored[0][0])
            best = scored[0]

        best_score = best[0] if best else 0.0
        best_c = best[1] if best else {}
        alts = scored[1:3]

        def alt_field(idx, key):
            if idx < len(alts):
                return alts[idx][1].get(key, "")
            return ""

        writer.writerow([
            row["LID"], artist, row.get("English", ""), row.get("Korean", ""), row.get("Genre", ""),
            query, decision, f"{best_score:.3f}",
            best_c.get("title", ""), best_c.get("channel") or best_c.get("uploader", ""),
            best_c.get("duration", ""), best_c.get("url", ""),
            alt_field(0, "title"), alt_field(0, "url"),
            alt_field(1, "title"), alt_field(1, "url"),
        ])
        out_f.flush()
        counts[decision] = counts.get(decision, 0) + 1
        print(f"[{i+1}/{len(rows)}] {row['LID']}: {decision} ({best_score:.2f})", file=sys.stderr)
        time.sleep(0.5)

    out_f.close()
    print(counts)


if __name__ == "__main__":
    main()
