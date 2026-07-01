#!/usr/bin/env python3
"""Map each OCR'd Mandarin lyric line onto the Korean-original timeline.

For every aligned cover we have:
  - ocr/lyrics/<idx>.jsonl : Mandarin lines with start/end in the COVER (ZH) timeline
  - alignments/<idx>.npz    : dense DTW warping path (cover_sec <-> orig_sec)

We interpolate each line's ZH start/end through the warping path to get the
corresponding Korean (KR) start/end. Output:
  - ocr/lyrics_aligned/<idx>.csv  (per song)
  - lyrics_aligned.csv  and  lyrics_aligned.jsonl  (combined, one row per line)

Run:  conda run -n music-translation python map_lyrics_to_kr.py
"""
import os, csv, json, glob
import numpy as np

BASE = os.environ.get("KPOP_DATA", "data")   # dataset root (set KPOP_DATA to your copy)
LYR = os.path.join(BASE, "ocr", "lyrics")
ALIGN = os.path.join(BASE, "alignments")
OUTDIR = os.path.join(BASE, "ocr", "lyrics_aligned")

def song_names():
    p = os.path.join(ALIGN, "alignment_summary.csv")
    return {int(r["idx"]): r["song"] for r in csv.DictReader(open(p))}

def warp_fn(idx):
    """Return (cover_sec, orig_sec) monotonic arrays for np.interp, plus matched span."""
    z = np.load(os.path.join(ALIGN, f"{idx:03d}.npz"), allow_pickle=True)
    cs, os_ = np.asarray(z["cover_sec"], float), np.asarray(z["orig_sec"], float)
    order = np.argsort(cs, kind="stable"); cs, os_ = cs[order], os_[order]
    # collapse duplicate cover times (average orig) so xp is strictly increasing
    uc, inv = np.unique(cs, return_inverse=True)
    uo = np.zeros_like(uc)
    cnt = np.zeros_like(uc)
    np.add.at(uo, inv, os_); np.add.at(cnt, inv, 1.0)
    uo /= np.maximum(cnt, 1)
    return uc, uo, (float(cs.min()), float(cs.max()))

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    names = song_names()
    combined = []
    for jf in sorted(glob.glob(os.path.join(LYR, "*.jsonl"))):
        idx = int(os.path.basename(jf)[:3])
        if not os.path.exists(os.path.join(ALIGN, f"{idx:03d}.npz")):
            continue
        cs, os_, (c0, c1) = warp_fn(idx)
        rows = []
        for i, line in enumerate((json.loads(l) for l in open(jf) if l.strip()), 1):
            zs, ze = line["start"], line["end"]
            kr_s = float(np.interp(zs, cs, os_))
            kr_e = float(np.interp(ze, cs, os_))
            in_span = (zs >= c0 - 0.5) and (ze <= c1 + 0.5)
            rows.append({"idx": idx, "song": names.get(idx, "?"), "line": i,
                         "zh_text": line["text"],
                         "zh_start": round(zs, 2), "zh_end": round(ze, 2),
                         "kr_start": round(kr_s, 2), "kr_end": round(kr_e, 2),
                         "conf": line.get("conf"), "in_aligned_span": in_span})
        with open(os.path.join(OUTDIR, f"{idx:03d}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())) if rows else None
            if w:
                w.writeheader(); w.writerows(rows)
        combined.extend(rows)

    if combined:
        with open(os.path.join(BASE, "lyrics_aligned.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(combined[0].keys()))
            w.writeheader(); w.writerows(combined)
        with open(os.path.join(BASE, "lyrics_aligned.jsonl"), "w") as f:
            for r in combined:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_songs = len({r["idx"] for r in combined})
    print(f"mapped {len(combined)} Mandarin lines across {n_songs} songs")
    print(f"-> {BASE}/lyrics_aligned.csv (+ .jsonl), per-song in {OUTDIR}/")

if __name__ == "__main__":
    main()
