#!/usr/bin/env python3
"""Join the forced-aligned Korean lyric lines to the Mandarin cover lines into a
one-to-one Korean<->Chinese bitext, enriched with script/romanization/translation.

Inputs (per song idx):
  kr_lyrics/<idx>.jsonl            Korean original lines + timings (align_kr_lyrics.py)
  kr_lyrics/<idx>.en.txt           parallel English translation, one line per Korean line
  ocr/lyrics_aligned/<idx>.csv     Mandarin cover lines with kr_start/kr_end (map_lyrics_to_kr.py)

Both the Korean lines and the cover lines live on the KOREAN timeline, so we join by
time overlap: each cover line is assigned to the Korean line it overlaps most, and the
cover lines under each Korean line are concatenated in order.

Per aligned row we emit:
  korean            Korean original line (Hangul, English kept as-is — the song mixes both)
  korean_rr         Revised Romanization of the Hangul (Latin runs pass through)
  english           English translation (from the parallel .en.txt)
  zh_hans           Mandarin cover line(s) — Simplified (OCR'd)
  zh_hant           Traditional Chinese (OpenCC s2twp: Taiwan standard + idioms)
  kr_start/kr_end   Korean-timeline span (s)
  zh_start/zh_end   cover-timeline span (s)

Outputs:  bitext/<idx>.csv (+ combined bitext.csv / bitext.jsonl)

Run:  conda run -n music-translation python build_bitext.py --idx 103
      conda run -n music-translation python build_bitext.py --all
"""
import os, csv, json, re, glob, argparse
from opencc import OpenCC
from korean_romanizer.romanizer import Romanizer

BASE = os.environ.get("KPOP_DATA", "data")   # dataset root (set KPOP_DATA to your copy)
KR = os.path.join(BASE, "kr_lyrics")
ALIGNED = os.path.join(BASE, "ocr", "lyrics_aligned")
OUT = os.path.join(BASE, "bitext")

_s2t = OpenCC("s2twp")
def to_traditional(text):
    return _s2t.convert(text)
def to_rr(text):                         # romanize Hangul runs, leave Latin/punct as-is
    return re.sub(r"[가-힣]+", lambda m: Romanizer(m.group()).romanize(), text)
def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))

def build(idx):
    kr = [json.loads(l) for l in open(os.path.join(KR, f"{idx:03d}.jsonl")) if l.strip()]
    en_path = os.path.join(KR, f"{idx:03d}.en.txt")
    en = [l.rstrip("\n") for l in open(en_path)] if os.path.exists(en_path) else [""] * len(kr)
    zh = list(csv.DictReader(open(os.path.join(ALIGNED, f"{idx:03d}.csv"))))
    for z in zh:
        z["kr_start"], z["kr_end"] = float(z["kr_start"]), float(z["kr_end"])
        z["zh_start"], z["zh_end"] = float(z["zh_start"]), float(z["zh_end"])

    # assign each cover line to the Korean line it overlaps most (>0)
    for z in zh:
        best, best_ov = None, 0.0
        for k in kr:
            if k["start"] is None:
                continue
            ov = _overlap(z["kr_start"], z["kr_end"], k["start"], k["end"])
            if ov > best_ov:
                best_ov, best = ov, k["line"]
        z["kr_line"] = best

    rows = []
    for k, en_line in zip(kr, en + [""] * (len(kr) - len(en))):
        mapped = sorted([z for z in zh if z["kr_line"] == k["line"]], key=lambda z: z["zh_start"])
        zh_hans = " ".join(z["zh_text"] for z in mapped)
        rows.append({
            "idx": idx, "kr_line": k["line"],
            "kr_start": k["start"], "kr_end": k["end"],
            "zh_start": mapped[0]["zh_start"] if mapped else None,
            "zh_end": mapped[-1]["zh_end"] if mapped else None,
            "korean": k["text"], "korean_rr": to_rr(k["text"]),
            "zh_hans": zh_hans, "zh_hant": to_traditional(zh_hans),
            "english": en_line,
        })
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{idx:03d}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int)
    ap.add_argument("--all", action="store_true", help="every kr_lyrics/*.jsonl")
    args = ap.parse_args()
    if args.all:
        targets = sorted(int(os.path.basename(p)[:3]) for p in glob.glob(os.path.join(KR, "*.jsonl")))
    else:
        targets = [args.idx]
    combined = []
    for i in targets:
        try:
            combined += build(i)
            print(f"[{i:03d}] {sum(1 for r in combined if r['idx']==i)} aligned lines", flush=True)
        except Exception as e:
            print(f"[{i:03d}] ERROR {type(e).__name__}: {e}", flush=True)
    if combined:
        with open(os.path.join(BASE, "bitext.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(combined[0].keys())); w.writeheader(); w.writerows(combined)
        with open(os.path.join(BASE, "bitext.jsonl"), "w") as f:
            for r in combined:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"-> {BASE}/bitext.csv (+ .jsonl), per-song in {OUT}/")
