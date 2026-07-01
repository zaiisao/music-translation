#!/usr/bin/env python3
"""Extract Mandarin 填词 (fan-cover) lyrics from the Bilibili cover videos with a
vision-LLM (Qwen3-VL).

Traditional OCR (EasyOCR, PaddleOCR) failed on these covers' stylized/outlined
subtitle fonts. Qwen3-VL reads them accurately and, via the prompt, returns ONLY the
lyric line — ignoring the uploader watermark, channel name, and credit list — which
also removes the need for a separate overlay filter.

Pipeline per video: download (Bilibili DASH web API, curl 412-workaround) -> sample
frames (ffmpeg) -> per-frame Qwen3-VL "read the lyric or NONE" -> dedup consecutive
frames into timed lines (COVER timeline). Outputs ocr/lyrics/<idx>.jsonl and .lrc.

Data directory is $KPOP_DATA (default below). Run (shard across GPUs):
    CUDA_VISIBLE_DEVICES=0 conda run -n qwen python vlm_lyrics.py --shard 0/2
    CUDA_VISIBLE_DEVICES=1 conda run -n qwen python vlm_lyrics.py --shard 1/2
"""
import os, csv, json, re, glob, shutil, argparse, difflib, subprocess
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

# ---------- paths ----------
BASE = os.environ.get("KPOP_DATA", "data")   # dataset root (set KPOP_DATA to your copy)
WORK = os.path.join(BASE, "ocr")
VID, FRM, LYR, RAW = (os.path.join(WORK, d) for d in ("video", "frames", "lyrics", "vlm_raw"))
COOKIES = os.path.join(WORK, "cookies.txt")
META = os.path.join(BASE, "covers_meta.json")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
REFERER = "Referer: https://www.bilibili.com/"

# ---------- Bilibili DASH download + frame sampling ----------
def ensure_cookies():
    os.makedirs(WORK, exist_ok=True)
    if os.path.exists(COOKIES) and os.path.getsize(COOKIES) > 0:
        return
    subprocess.run(["curl", "-s", "-c", COOKIES, "-A", UA, "https://www.bilibili.com/", "-o", os.devnull])

def curl_json(url):
    out = subprocess.run(["curl", "-s", "-b", COOKIES, "-A", UA, "-H", REFERER, url],
                         capture_output=True, text=True).stdout
    return json.loads(out)

def content_length(url):
    r = subprocess.run(["curl", "-sI", "-L", "--http1.1", "-A", UA, "-H", REFERER, url],
                       capture_output=True, text=True)
    for ln in r.stdout.splitlines():
        if ln.lower().startswith("content-length:"):
            try: return int(ln.split(":", 1)[1].strip())
            except Exception: pass
    return None

def download_video(bv, cid, dst, max_h=720):
    """Fetch a DASH video stream (prefer H.264 <= max_h). Clean per-mirror download —
    no cross-mirror `-C -` resume, which corrupts the h264 stream mid-file."""
    if os.path.exists(dst) and os.path.getsize(dst) > 100000:
        return True, "cached"
    pu = curl_json(f"https://api.bilibili.com/x/player/playurl?bvid={bv}&cid={cid}&qn=0&fnval=16&fourk=1")
    if pu.get("code") != 0:
        return False, f"playurl err {pu.get('code')}"
    vids = pu["data"].get("dash", {}).get("video", []) or []
    if not vids:
        return False, "no dash video"
    avc = [v for v in vids if str(v.get("codecs", "")).startswith("avc")] or vids
    le = [v for v in avc if (v.get("height") or 0) <= max_h]
    pick = max(le, key=lambda v: v.get("height", 0)) if le else min(avc, key=lambda v: v.get("height", 1e9))
    urls = []
    for u in [pick.get("baseUrl")] + (pick.get("backupUrl") or []):
        if u and u not in urls: urls.append(u)
    tmp = dst + ".part"
    for u in urls:
        exp = content_length(u)
        for _ in range(3):
            if os.path.exists(tmp): os.remove(tmp)
            subprocess.run(["curl", "-s", "-L", "--http1.1", "-A", UA, "-H", REFERER,
                            "--retry", "5", "--retry-delay", "2", "-o", tmp, u])
            sz = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            if (exp and sz == exp) or (not exp and sz > 100000):
                break
        if os.path.exists(tmp) and (not exp or os.path.getsize(tmp) == exp):
            shutil.move(tmp, dst)
            return True, f"{pick.get('height')}p {os.path.getsize(dst)//1024}KB codec={pick.get('codecs')}"
    if os.path.exists(tmp): os.remove(tmp)
    return False, "download failed"

def extract_frames(video, out_dir, fps):
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "*.jpg")): os.remove(f)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-err_detect", "ignore_err",
                    "-fflags", "+discardcorrupt", "-i", video,
                    "-vf", f"fps={fps}", "-q:v", "3", os.path.join(out_dir, "f%05d.jpg")])
    return sorted(glob.glob(os.path.join(out_dir, "*.jpg")))

def load_meta():
    return {r["idx"]: r for r in json.load(open(META)) if "bv" in r}

def aligned_indices():
    """Cover indices that were aligned to a Korean original."""
    p = os.path.join(BASE, "alignments", "alignment_summary.csv")
    return sorted(int(r["idx"]) for r in csv.DictReader(open(p)))

# ---------- Qwen3-VL ----------
MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
PROMPT = ("只输出画面中的中文歌词字幕文字。忽略UP主水印、频道名（如'杰的冰美式JeJe'、'bilibili'）、"
          "以及'原唱/填词/翻唱/视封/混音'等制作人员名单信息。如果画面没有歌词，只输出 NONE。"
          "只输出歌词文字本身（多行用空格连接），不要输出任何解释。")

_M = _P = None
def load():
    global _M, _P
    if _M is None:
        _M = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype="auto", device_map="auto")
        _P = AutoProcessor.from_pretrained(MODEL)
    return _M, _P

def infer_batch(paths):
    m, p = load()
    msgs = [[{"role": "user", "content": [{"type": "image", "image": x},
                                          {"type": "text", "text": PROMPT}]}] for x in paths]
    inp = p.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                return_dict=True, return_tensors="pt", padding=True).to(m.device)
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=48, do_sample=False)
    L = inp["input_ids"].shape[1]
    return [p.decode(out[i][L:], skip_special_tokens=True).strip() for i in range(len(paths))]

# ---------- build timed lyric lines ----------
def _norm(t):
    return re.sub(r"\s+", "", re.sub(r"[，。！？、,.!?~\-—…·:：\"'’”“()（）\d]", "", t))
def _is_none(t):
    return (t.strip().upper() == "NONE") or (len(_norm(t)) == 0)

# Intro title-card / watermark credit lines the VLM sometimes reads despite the prompt,
# e.g. "原唱 | Jin", "翻唱&视封 | JeJe", "填词 | Yizy尹姿", "混音:太咸", "原唱 Irene", "翻唱：".
_CREDIT_KW = re.compile(r"(原唱|翻唱|填词|作词|混音|视封|初音|编曲|后期|监制|策划|美工|海报|字幕|翻译|封面)")
_SEP = re.compile(r"[|｜:：]")
_CREDIT_LEAD = re.compile(r"^\s*(原唱|翻唱|填词|作词|混音|视封|初音|编曲|后期|监制|策划|美工|海报|字幕|翻译|封面)[\s:：|｜]")
_CREDIT_TRAIL = re.compile(r"[\s:：|｜](原唱|翻唱|填词|作词|混音|视封|初音|编曲|后期|监制)\s*$")
def _is_credit(txt):
    if "jeje" in txt.lower():                       # uploader handle (watermark/credits)
        return True
    if _CREDIT_KW.search(txt) and _SEP.search(txt): # "翻唱：JeJe", "混音:太咸"
        return True
    if _CREDIT_LEAD.match(txt):                     # "混音 小鱼", "原唱 Irene", "填词 尹姿"
        return True
    if _CREDIT_TRAIL.search(txt):                   # "麦酒酿 填词", "太咸 混音"
        return True
    if _CREDIT_KW.fullmatch(_norm(txt)):            # bare label: "翻唱", "填词"
        return True
    return False

# Intro title-card / version label, e.g. "Lemonade 中文版 (Chinese Ver.)", "Cosmic 中文版",
# "… 中文填词翻唱", "升3key版翻唱 纯人声版". These lead the clip; strip the label span and
# keep any real lyric that got merged onto the same frame ("… 中文版 You just gotta let me go").
_TITLECARD = re.compile(
    r"^.*?(中文版(\s*[（(]\s*chinese\s*ver\.?\s*[)）])?|[（(]\s*chinese\s*ver\.?\s*[)）]|"
    r"中文填词翻唱|填词翻唱|纯人声版|纯人声|升\s*\d*\s*key\S*)\s*",
    re.I)
def _strip_titlecard(txt):
    return _TITLECARD.sub("", txt, count=1)

def build_lines(idx, fps, texts, merge_sim=0.6):
    merged = []
    for k, txt in enumerate(texts):
        t = round(k / fps, 2)
        if _is_none(txt) or _is_credit(txt):
            continue
        txt = _strip_titlecard(txt)                 # drop leading "<title> 中文版 …" label
        txt = re.sub(r"\s+", " ", txt.replace("\n", " ")).strip()
        if len(_norm(txt)) < 2:
            continue
        key = _norm(txt)
        if merged:
            pk = _norm(merged[-1]["text"])
            if key == pk or key in pk or pk in key or difflib.SequenceMatcher(None, key, pk).ratio() >= merge_sim:
                merged[-1]["end"] = t
                if len(txt) > len(merged[-1]["text"]):
                    merged[-1]["text"] = txt
                continue
        merged.append({"start": t, "end": t, "text": txt})
    with open(os.path.join(LYR, f"{idx:03d}.jsonl"), "w") as fh:
        for m in merged:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(os.path.join(LYR, f"{idx:03d}.lrc"), "w") as fh:
        for m in merged:
            mm, ss = divmod(m["start"], 60)
            fh.write(f"[{int(mm):02d}:{ss:05.2f}] {m['text']}\n")
    return len(merged)

def rebuild(idx, merge_sim=0.6):
    d = json.load(open(os.path.join(RAW, f"{idx:03d}.json")))
    return {"idx": idx, "ok": True, "n_lines": build_lines(idx, d["fps"], d["texts"], merge_sim)}

def process(idx, bv, cid, fps=2.0, batch=8, merge_sim=0.6, keep_frames=False):
    for d in (VID, LYR, RAW): os.makedirs(d, exist_ok=True)
    vpath = os.path.join(VID, f"{idx:03d}.mp4")
    ok, msg = download_video(bv, cid, vpath)
    if not ok:
        return {"idx": idx, "ok": False, "msg": msg}
    fdir = os.path.join(FRM, f"{idx:03d}")
    frames = extract_frames(vpath, fdir, fps)
    texts = []
    for i in range(0, len(frames), batch):
        texts.extend(infer_batch(frames[i:i + batch]))
    json.dump({"idx": idx, "fps": fps, "texts": texts},
              open(os.path.join(RAW, f"{idx:03d}.json"), "w"), ensure_ascii=False)
    n = build_lines(idx, fps, texts, merge_sim)
    if not keep_frames:
        shutil.rmtree(fdir, ignore_errors=True)
    return {"idx": idx, "ok": True, "n_frames": len(frames), "n_lines": n, "video": msg}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int)
    ap.add_argument("--all", action="store_true", help="all covers (default: the aligned ones)")
    ap.add_argument("--shard", type=str, default=None, help="i/n : every n-th target (multi-GPU)")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--merge-sim", type=float, default=0.6)
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="rebuild lines from cached VLM output (no model)")
    args = ap.parse_args()
    meta = load_meta()
    if args.idx:
        targets = [args.idx]
    elif args.all:
        targets = sorted(meta)
    else:
        targets = aligned_indices()
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        targets = targets[i::n]
    if not args.rebuild:
        ensure_cookies()
    for i in targets:
        try:
            if args.rebuild:
                res = rebuild(i, merge_sim=args.merge_sim)
            else:
                res = process(i, meta[i]["bv"], meta[i]["cid"], fps=args.fps,
                              batch=args.batch, merge_sim=args.merge_sim, keep_frames=args.keep_frames)
        except Exception as e:
            res = {"idx": i, "ok": False, "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(res, ensure_ascii=False), flush=True)
