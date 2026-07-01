#!/usr/bin/env python3
"""Forced-align the Korean original lyrics to the Korean audio -> per-line timings.

Input : kr_lyrics/<idx>.txt   (official Korean lyrics, one line per line, in order;
                                English lines kept as-is — the song mixes languages)
Output: kr_lyrics/<idx>.jsonl (one row per line: {line, start, end, text})

Uses ctc-forced-aligner (multilingual MMS CTC + uroman). The library aligns at the
WORD level; since preprocess_text() splits on whitespace 1:1 before romanizing, the
returned word_timestamps line up exactly with text.split(), so we regroup words back
into the original lines by word count (the library's own line remapper compares
romanized text to Hangul and fails, so we do it ourselves).

Run (needs GPU env with ctc-forced-aligner):
    conda run -n qwen python align_kr_lyrics.py --idx 103
    conda run -n qwen python align_kr_lyrics.py --all
"""
import os, csv, json, argparse
import onnxruntime
import ctc_forced_aligner as fa

BASE = os.environ.get("KPOP_DATA", "data")   # dataset root (set KPOP_DATA to your copy)
LYR = os.path.join(BASE, "kr_lyrics")
MODEL = os.path.join(BASE, "models", "ctc_forced_aligner.onnx")
LANG = "kor"

_SESS = _TOK = None
def _load():
    global _SESS, _TOK
    if _SESS is None:
        fa.ensure_onnx_model(MODEL, fa.MODEL_URL)
        _SESS = onnxruntime.InferenceSession(
            MODEL, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        _TOK = fa.Tokenizer()
    return _SESS, _TOK

def original_path(idx):
    with open(os.path.join(BASE, "manifest.csv")) as f:
        for r in csv.DictReader(f):
            if int(r["idx"]) == idx:
                return os.path.join(BASE, "originals", r["original_file"])
    raise KeyError(idx)

def align(idx):
    sess, tok = _load()
    lines = [l.strip() for l in open(os.path.join(LYR, f"{idx:03d}.txt")) if l.strip()]
    audio = fa.load_audio(original_path(idx))                      # 16 kHz mono
    emissions, stride = fa.generate_emissions(sess, audio, batch_size=4)
    tokens_starred, text_starred = fa.preprocess_text(" ".join(lines), romanize=True, language=LANG)
    segments, scores, blank = fa.get_alignments(emissions, tokens_starred, tok)
    spans = fa.get_spans(tokens_starred, segments, blank)
    words = fa.postprocess_results(text_starred, spans, stride, scores)

    rows, i = [], 0
    for k, line in enumerate(lines, 1):
        n = len(line.split())
        chunk = words[i:i + n]; i += n
        rows.append({"line": k,
                     "start": round(chunk[0]["start"], 2) if chunk else None,
                     "end": round(chunk[-1]["end"], 2) if chunk else None,
                     "text": line})
    out = os.path.join(LYR, f"{idx:03d}.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"idx": idx, "lines": len(rows), "words": len(words), "out": out}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int)
    ap.add_argument("--all", action="store_true", help="every kr_lyrics/*.txt")
    args = ap.parse_args()
    if args.all:
        import glob
        targets = sorted(int(os.path.basename(p)[:3]) for p in glob.glob(os.path.join(LYR, "*.txt")))
    else:
        targets = [args.idx]
    for i in targets:
        try:
            print(json.dumps(align(i), ensure_ascii=False), flush=True)
        except Exception as e:
            print(json.dumps({"idx": i, "error": f"{type(e).__name__}: {e}"}), flush=True)
